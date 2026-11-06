import array
import os
import struct
from pysol import (
    Array, KernelArgument, Type, import_kernel, pysol,
    pysol_compile, pysol_run, load_config, pysol_update_array,
)

# Mirrors the block/grid literals inside lattice_gas() below - the @pysol
# AST analysis only understands literals local to the decorated function,
# so write_vtk_file() (the Python-side post-processing step, at the bottom
# of this file) needs its own copies to know the grid/block layout.
NB, MB = 3, 3
BN, BM = 667, 134
NX, NY = NB * BN, MB * BM
NUM_TS = 100000
DUMP_EVERY = 1000  # must match the "ts % 1000" check in dump_lbm_block

# =============================================================================
# pysol reimplementation of examples/lattice-gas/lattice-gas.cpp's fluid
# solver (D2Q9 BGK lattice-Boltzmann) - fluid only, as a lid-driven cavity
# instead of the reference's channel + immersed-boundary rigid particle.
#
# The reference couples its grid to a separate set of 160 scattered
# immersed-boundary points (a circle) via force spreading and velocity
# interpolation. That coupling is a gather/scatter between arbitrary,
# continuously-moving points and whichever grid blocks happen to be near
# them at a given instant, which doesn't fit pysol's block-local kernel
# model the way a fixed, structured field does - so this port keeps only
# the grid fluid solver (collide_bulk/stream/update_bounce_back) and
# drops the immersed boundary entirely: no points/node_struct, no
# per-particle force, no fx/fy force field (force_x=force_y=0.0 in the
# reference once the IB particle is removed, so the force terms in
# collide_bulk's feq/force computation vanish and are dropped here too).
# In their place, the top wall carries a prescribed tangential velocity
# (the lid), which is what actually drives the flow in this port - the
# reference's own vel_center plays the same role as an inlet velocity in
# its channel.
#
# EVERYTHING is fragmented into an nb x mb block grid, mirroring the
# pic3d port's structure:
# - f0..f8 (pre-collision populations): no ghost - collision only reads
#   a node's own populations, never a neighbor's.
# - f2_0..f2_8 (post-collision/post-bounce-back populations): 1-node
#   ghost layer, ghost-exchanged via pysol_update_array once per
#   timestep, exactly like pic3d's ef_x/ef_y/ef_z, before stream_block
#   reads a neighboring block's edge.
# - rho/ux/uy: fragmented, no-ghost diagnostic fields recomputed by
#   collide_block every timestep.
# None of this needs a GLOBAL, non-fragmented array or a static C++
# buffer anywhere (unlike the reference's file-scope f/f2/ux/uy/rho).
#
# Struct types aren't supported for non-growing (plain Array) arrays, and
# this port has no GrowingArray use case (every field here is a fixed-size
# per-node value, never a variable-length collection) - which is also why
# the IB points (naturally a struct-per-point in the reference) are one
# more reason to drop them rather than force them into this model. Every
# array below is kept at n_dims<=2 - the population index (0..8) is NOT a
# third array dimension; it's 9 separate 2D arrays (f0..f8, f2_0..f2_8)
# instead, enumerated explicitly at each call site the same way pic3d
# enumerates ef_x/ef_y/ef_z rather than using a 4th array dimension.
# =============================================================================


@pysol
def lattice_gas():
    # ── kernel imports ────────────────────────────────────────────────────────
    init_lbm_k = import_kernel("init_lbm_block",
                                [KernelArgument("bn", True, Type.Int),
                                 KernelArgument("bm", True, Type.Int),
                                 KernelArgument("f0", False, Type.Double, 2),
                                 KernelArgument("f1", False, Type.Double, 2),
                                 KernelArgument("f2", False, Type.Double, 2),
                                 KernelArgument("f3", False, Type.Double, 2),
                                 KernelArgument("f4", False, Type.Double, 2),
                                 KernelArgument("f5", False, Type.Double, 2),
                                 KernelArgument("f6", False, Type.Double, 2),
                                 KernelArgument("f7", False, Type.Double, 2),
                                 KernelArgument("f8", False, Type.Double, 2)])

    # Block (ii,jj)'s sole producer of its own next-version f2/rho/ux/uy -
    # see its long comment in ucodes.h for why collision and bounce-back
    # are combined into one kernel (single-assignment) instead of the
    # reference's two separate passes over f2.
    collide_k = import_kernel("collide_block",
                               [KernelArgument("ii", True, Type.Int),
                                KernelArgument("jj", True, Type.Int),
                                KernelArgument("bn", True, Type.Int),
                                KernelArgument("bm", True, Type.Int),
                                KernelArgument("gnx", True, Type.Int),
                                KernelArgument("gny", True, Type.Int),
                                KernelArgument("omega", True, Type.Double),
                                KernelArgument("vel_lid", True, Type.Double),
                                KernelArgument("f0", True, Type.Double, 2),
                                KernelArgument("f1", True, Type.Double, 2),
                                KernelArgument("f2", True, Type.Double, 2),
                                KernelArgument("f3", True, Type.Double, 2),
                                KernelArgument("f4", True, Type.Double, 2),
                                KernelArgument("f5", True, Type.Double, 2),
                                KernelArgument("f6", True, Type.Double, 2),
                                KernelArgument("f7", True, Type.Double, 2),
                                KernelArgument("f8", True, Type.Double, 2),
                                KernelArgument("f2_0", False, Type.Double, 2),
                                KernelArgument("f2_1", False, Type.Double, 2),
                                KernelArgument("f2_2", False, Type.Double, 2),
                                KernelArgument("f2_3", False, Type.Double, 2),
                                KernelArgument("f2_4", False, Type.Double, 2),
                                KernelArgument("f2_5", False, Type.Double, 2),
                                KernelArgument("f2_6", False, Type.Double, 2),
                                KernelArgument("f2_7", False, Type.Double, 2),
                                KernelArgument("f2_8", False, Type.Double, 2),
                                KernelArgument("rho", False, Type.Double, 2),
                                KernelArgument("ux", False, Type.Double, 2),
                                KernelArgument("uy", False, Type.Double, 2)])

    # Reads the ghost-exchanged f2 (see the pysol_update_array calls in
    # the main loop below) to write this block's fresh f for the next
    # timestep - mirrors compute_efield_block's post-exchange read in the
    # pic3d port.
    stream_k = import_kernel("stream_block",
                              [KernelArgument("bn", True, Type.Int),
                               KernelArgument("bm", True, Type.Int),
                               KernelArgument("f2_0", True, Type.Double, 2),
                               KernelArgument("f2_1", True, Type.Double, 2),
                               KernelArgument("f2_2", True, Type.Double, 2),
                               KernelArgument("f2_3", True, Type.Double, 2),
                               KernelArgument("f2_4", True, Type.Double, 2),
                               KernelArgument("f2_5", True, Type.Double, 2),
                               KernelArgument("f2_6", True, Type.Double, 2),
                               KernelArgument("f2_7", True, Type.Double, 2),
                               KernelArgument("f2_8", True, Type.Double, 2),
                               KernelArgument("f0", False, Type.Double, 2),
                               KernelArgument("f1", False, Type.Double, 2),
                               KernelArgument("f2", False, Type.Double, 2),
                               KernelArgument("f3", False, Type.Double, 2),
                               KernelArgument("f4", False, Type.Double, 2),
                               KernelArgument("f5", False, Type.Double, 2),
                               KernelArgument("f6", False, Type.Double, 2),
                               KernelArgument("f7", False, Type.Double, 2),
                               KernelArgument("f8", False, Type.Double, 2)])

    # Writes block (ii,jj)'s own rho/ux/uy straight to its own file under
    # results/ - no in-program gather, so this stays MPI-safe the same
    # way pic3d's dump_vti_block does. write_vtk_file() below (plain
    # Python, run after the simulation) reads every block's file back and
    # assembles a single VTK file. The "every 10th step" gate lives
    # inside dump_lbm_block itself (plain "if (ts % 10 != 0) return;" in
    # C++), not as a pysol `if` here - this example's arrays are all
    # n_dims=2, so sgc's if-with-array-argument restriction (which only
    # bit pic3d's 3D arrays) likely wouldn't apply, but gating inside the
    # kernel is simpler and keeps the call site unconditional either way.
    dump_k = import_kernel("dump_lbm_block",
                            [KernelArgument("ts", True, Type.Int),
                             KernelArgument("ii", True, Type.Int),
                             KernelArgument("jj", True, Type.Int),
                             KernelArgument("bn", True, Type.Int),
                             KernelArgument("bm", True, Type.Int),
                             KernelArgument("rho", True, Type.Double, 2),
                             KernelArgument("ux", True, Type.Double, 2),
                             KernelArgument("uy", True, Type.Double, 2)])

    print_step_k = import_kernel("print_step", [KernelArgument("ts", True, Type.Int)])

    # Streams block (ii,jj)'s velocity to visualizer.py over a TCP socket
    # as soon as it's produced, instead of waiting for a file dump - see
    # send_velocity_block's long comment in ucodes.h for the wire format
    # and the LATTICE_GAS_VIZ_* environment variables that configure the
    # host/port/cadence. A separate kernel rather than folding this into
    # dump_k: live viz wants a finer cadence than the file dump's "every
    # 1000th step" (see LATTICE_GAS_VIZ_EVERY), and keeping it separate
    # means dump_lbm_block's already-tested file-writing logic is
    # untouched.
    send_viz_k = import_kernel("send_velocity_block",
                                [KernelArgument("ts", True, Type.Int),
                                 KernelArgument("ii", True, Type.Int),
                                 KernelArgument("jj", True, Type.Int),
                                 KernelArgument("bn", True, Type.Int),
                                 KernelArgument("bm", True, Type.Int),
                                 KernelArgument("gnx", True, Type.Int),
                                 KernelArgument("gny", True, Type.Int),
                                 KernelArgument("ux", True, Type.Double, 2),
                                 KernelArgument("uy", True, Type.Double, 2)])

    # ── grid/block parameters ──────────────────────────────────────────────
    # NX=2001, NY=402 matches the reference's init_hydro() exactly;
    # 2001=3*667 and 402=3*134 factor cleanly into a 3x3 block grid.
    nb : Type.Int = 3
    mb : Type.Int = 3
    bn : Type.Int = 667
    bm : Type.Int = 134
    gnx : Type.Int = nb * bn
    gny : Type.Int = mb * bm
    # Matches the reference's N=100000 main-loop bound exactly (its
    # "for (counter=0; counter<=N; counter++)" is N+1 iterations, hence
    # "while step <= num_ts:" below rather than "<").
    num_ts : Type.Int = 100000

    # Lid (top wall) velocity and the matching BGK relaxation parameter -
    # same formula the reference uses for vel_center/omega, with vel_lid
    # playing vel_center's role as the flow driver.
    vel_lid : Type.Double = 0.05
    omega : Type.Double = 1.0 / (0.5 + vel_lid / 0.1 * (1.1 - 0.5))

    # ── arrays ───────────────────────────────────────────────────────────────
    f0 = Array(2, 2, [nb, mb], [bn, bm], Type.Double)
    f1 = Array(2, 2, [nb, mb], [bn, bm], Type.Double)
    f2 = Array(2, 2, [nb, mb], [bn, bm], Type.Double)
    f3 = Array(2, 2, [nb, mb], [bn, bm], Type.Double)
    f4 = Array(2, 2, [nb, mb], [bn, bm], Type.Double)
    f5 = Array(2, 2, [nb, mb], [bn, bm], Type.Double)
    f6 = Array(2, 2, [nb, mb], [bn, bm], Type.Double)
    f7 = Array(2, 2, [nb, mb], [bn, bm], Type.Double)
    f8 = Array(2, 2, [nb, mb], [bn, bm], Type.Double)

    f2_0 = Array(2, 2, [nb, mb], [bn + 2, bm + 2], Type.Double)
    f2_1 = Array(2, 2, [nb, mb], [bn + 2, bm + 2], Type.Double)
    f2_2 = Array(2, 2, [nb, mb], [bn + 2, bm + 2], Type.Double)
    f2_3 = Array(2, 2, [nb, mb], [bn + 2, bm + 2], Type.Double)
    f2_4 = Array(2, 2, [nb, mb], [bn + 2, bm + 2], Type.Double)
    f2_5 = Array(2, 2, [nb, mb], [bn + 2, bm + 2], Type.Double)
    f2_6 = Array(2, 2, [nb, mb], [bn + 2, bm + 2], Type.Double)
    f2_7 = Array(2, 2, [nb, mb], [bn + 2, bm + 2], Type.Double)
    f2_8 = Array(2, 2, [nb, mb], [bn + 2, bm + 2], Type.Double)

    rho = Array(2, 2, [nb, mb], [bn, bm], Type.Double)
    ux = Array(2, 2, [nb, mb], [bn, bm], Type.Double)
    uy = Array(2, 2, [nb, mb], [bn, bm], Type.Double)

    # ── initial setup (rest state everywhere) ───────────────────────────────
    for i in range(nb):
        for j in range(mb):
            init_lbm_k(bn, bm, f0[i][j], f1[i][j], f2[i][j], f3[i][j], f4[i][j],
                       f5[i][j], f6[i][j], f7[i][j], f8[i][j])

    # ── main LBM loop ────────────────────────────────────────────────────────
    # A `while` loop here, not "for ts in range(num_ts):" - confirmed by a
    # minimal standalone reproduction that a `for ts in range(N):` outer
    # loop wrapping a kernel call on a *2D* array hangs after the first
    # outer iteration (the kernel never runs again), while the identical
    # logic with an equivalent `while` loop runs correctly. 2D (and
    # presumably 1D) arrays are code-generated through a different,
    # simpler "plain task" path (rts::Task{...}/ref_tgp_in_var/
    # ref_tgp_out_var in the generated .cpp) than 3D arrays' ESS/stencil
    # path (essp_N->insert_task, what pic3d's "for ts in range(num_ts):"
    # main loop relies on without issue) - this hang is specific to that
    # plain-task path, not to "for" loops in general.
    # "ts" is the array VERSION counter (each real timestep consumes 2
    # version numbers - one for collide_k's raw output, one for the
    # post-exchange copy - exactly like pic3d's phi "it"/"it+1"/"it+2"
    # ping-pong); "step" is the REAL timestep index, matching the
    # reference's "counter" 1:1, used only for print_step_k/dump_k (both
    # the "every NOUTPUT steps" gate inside dump_lbm_block and the output
    # filenames). Keeping them separate matters: ts is always even, so
    # using it directly for the gate/filenames would silently halve the
    # intended output period.
    ts : Type.Int = 0
    step : Type.Int = 0
    while step <= num_ts:
        for i in range(nb):
            for j in range(mb):
                collide_k(i, j, bn, bm, gnx, gny, omega, vel_lid,
                          f0[ts, i][j], f1[ts, i][j], f2[ts, i][j], f3[ts, i][j], f4[ts, i][j],
                          f5[ts, i][j], f6[ts, i][j], f7[ts, i][j], f8[ts, i][j],
                          f2_0[ts, i][j], f2_1[ts, i][j], f2_2[ts, i][j], f2_3[ts, i][j],
                          f2_4[ts, i][j], f2_5[ts, i][j], f2_6[ts, i][j], f2_7[ts, i][j],
                          f2_8[ts, i][j],
                          rho[ts, i][j], ux[ts, i][j], uy[ts, i][j])

        pysol_update_array(f2_0, ts, f2_0, ts + 1)
        pysol_update_array(f2_1, ts, f2_1, ts + 1)
        pysol_update_array(f2_2, ts, f2_2, ts + 1)
        pysol_update_array(f2_3, ts, f2_3, ts + 1)
        pysol_update_array(f2_4, ts, f2_4, ts + 1)
        pysol_update_array(f2_5, ts, f2_5, ts + 1)
        pysol_update_array(f2_6, ts, f2_6, ts + 1)
        pysol_update_array(f2_7, ts, f2_7, ts + 1)
        pysol_update_array(f2_8, ts, f2_8, ts + 1)

        for i in range(nb):
            for j in range(mb):
                stream_k(bn, bm,
                         f2_0[ts + 1, i][j], f2_1[ts + 1, i][j], f2_2[ts + 1, i][j], f2_3[ts + 1, i][j],
                         f2_4[ts + 1, i][j], f2_5[ts + 1, i][j], f2_6[ts + 1, i][j], f2_7[ts + 1, i][j],
                         f2_8[ts + 1, i][j],
                         f0[ts + 2, i][j], f1[ts + 2, i][j], f2[ts + 2, i][j], f3[ts + 2, i][j], f4[ts + 2, i][j],
                         f5[ts + 2, i][j], f6[ts + 2, i][j], f7[ts + 2, i][j], f8[ts + 2, i][j])

        print_step_k(step)

        for i in range(nb):
            for j in range(mb):
                dump_k(step, i, j, bn, bm, rho[ts, i][j], ux[ts, i][j], uy[ts, i][j])
                send_viz_k(step, i, j, bn, bm, gnx, gny, ux[ts, i][j], uy[ts, i][j])

        ts = ts + 2
        step = step + 1


def _flat2(i, j, m):
    return i * m + j


def _read_doubles(path, count):
    a = array.array("d")
    with open(path, "rb") as fh:
        a.frombytes(fh.read(count * 8))
    return a


def write_vtk_file():
    """Post-processing step, run after the simulation: for every dumped
    timestep, gathers every block's dump_lbm_block file and assembles
    results/lbm_<ts>.vtk (VTK STRUCTURED_POINTS, scalar density + vector
    velocity) - the fluid-only part of what the reference's writevtk()
    writes (dropping its _imm.vtk immersed-boundary output, since this
    port has no immersed boundary). Done here in plain Python instead of
    inside the .sol-compiled program, the same way pic3d's
    write_vti_files() reads per-block files back after the run rather
    than gathering them while it's still running - this is what keeps
    the dump MPI-safe, since each rank only ever writes the files for the
    blocks it owns.
    """
    n = NX * NY
    # NUM_TS+1: the simulation's "while step <= num_ts:" dumps at
    # step==NUM_TS too (NUM_TS % 1000 == 0), which plain range(0, NUM_TS,
    # DUMP_EVERY) would miss since range()'s stop bound is exclusive.
    for ts in range(0, NUM_TS + 1, DUMP_EVERY):
        rho = array.array("d", [0.0]) * n
        ux = array.array("d", [0.0]) * n
        uy = array.array("d", [0.0]) * n

        found_any = False
        for ii in range(NB):
            for jj in range(MB):
                block_path = f"results/lbm_block_{ts:05d}_{ii}_{jj}.bin"
                if not os.path.exists(block_path):
                    continue
                found_any = True
                rec = _read_doubles(block_path, 3 * BN * BM)
                idx = 0
                for li in range(BN):
                    for lj in range(BM):
                        gi, gj = ii * BN + li, jj * BM + lj
                        gidx = _flat2(gi, gj, NY)
                        rho[gidx] = rec[idx]
                        ux[gidx] = rec[idx + 1]
                        uy[gidx] = rec[idx + 2]
                        idx += 3

        if not found_any:
            continue

        with open(f"results/lbm_{ts:05d}.vtk", "w") as out:
            out.write("# vtk DataFile Version 3.0\n")
            out.write("Hydrodynamics representation\n")
            out.write("ASCII\n\n")
            out.write("DATASET STRUCTURED_POINTS\n")
            out.write(f"DIMENSIONS {NX} {NY} 1\n")
            out.write("ORIGIN 0 0 0\n")
            out.write("SPACING 1 1 1\n")
            out.write(f"POINT_DATA {NX * NY}\n")

            # VTK STRUCTURED_POINTS expects X varying fastest (matching
            # the reference's own writevtk(), whose counter increments
            # iX before iY) - the opposite order from _flat2's storage
            # layout above (which has Y as the fast-varying index), so
            # these loops are nested Y-outer, X-inner.
            out.write("SCALARS density double\n")
            out.write("LOOKUP_TABLE density_table\n")
            for gj in range(NY):
                for gi in range(NX):
                    out.write(f"{rho[_flat2(gi, gj, NY)]}\n")

            out.write("VECTORS velocity double\n")
            for gj in range(NY):
                for gi in range(NX):
                    n_ = _flat2(gi, gj, NY)
                    out.write(f"{ux[n_]} {uy[n_]} 0\n")


if __name__ == "__main__":
    load_config("config.json")

    print("==========LATTICE-GAS (pysol)============")
    pysol_compile("lattice_gas")
    pysol_run("lattice_gas")
    write_vtk_file()
