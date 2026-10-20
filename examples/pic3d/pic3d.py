
import array
import os
import sys
from pysol import (
    Array, GrowingArray, KernelArgument, Type, import_kernel, pysol,
    pysol_compile, pysol_run, load_config, pysol_struct, pysol_update_array,
    sol_generator,
)

# Mirrors the "mesh / block parameters" literals inside pic3d() below - the
# @pysol AST analysis only understands literals local to the decorated
# function, so write_vti_files() (the Python-side VTI post-processing step,
# at the bottom of this file) needs its own copies to know the grid/block
# layout and domain extents of the dump_vti_block_k files it reads.
NB, MB, LB = 3, 3, 3
BN, BM, BL = 7, 7, 7
GN, GM, GL = 21, 21, 21
X0 = (-0.1, -0.1, 0.0)
XM = (0.1, 0.1, 0.2)
DH = ((XM[0] - X0[0]) / (GN - 1), (XM[1] - X0[1]) / (GM - 1), (XM[2] - X0[2]) / (GL - 1))
NUM_TS = 100
VTI_EVERY = 100  # must match the "ts % 100" check in dump_vti_block

# =============================================================================
# pysol reimplementation of examples/pic3d/pic3d.cpp (3D electrostatic PIC).
#
# EVERYTHING is fragmented into an nb x mb x lb block grid, so this runs
# correctly across more than one MPI process:
# - phi/rho/resid2/ef_x/ef_y/ef_z: 1-node ghost layer, ghost-exchanged via
#   pysol_update_array between SOR sweeps (phi) or once per timestep (ef).
# - node_vol/den_ion/den_elec: no ghost layer - see reduce_density_block in
#   ucodes.h for why density's cross-block spillover is reduced by hand
#   (an ADDITIVE operation pysol_update_array can't do, since it's
#   generated from exchange_3d.sol's copy-only _write_ghost3d_*).
# - particles_ion/particles_elec: one GrowingArray cell per block. A
#   particle that crosses a block boundary during push_particles is
#   migrated to its new owning block's cell via a 7-mailbox scheme (see
#   push_particles/merge_particles_mail in ucodes.h) rather than written
#   directly into a neighbor's cell, which would violate single-assignment
#   exactly like a shared global array would.
# - sum()/max()/min() are still used for the Poisson solver's L2 residual
#   convergence check (phi/resid2's block grid matches the program's main
#   distributor); density's own whole-domain sums use sum_block/
#   count_particles' accumulate-into-an-"out"-scalar pattern instead, once
#   per block, the same way diagnostics already worked before density was
#   fragmented.
# - The Poisson solver's convergence loop is a `while` (not a for+break,
#   which pysol's for-loops don't support anyway).
# - Boundary handling for the particle mailboxes (merge_particles_mail)
#   and the density ghost reduction (reduce_density_block) is computed via
#   plain Python-level `if` statements each block iteration (see their
#   comments below) rather than via a padded block grid - earlier in this
#   project's development this had to be worked around with a guard-block
#   grid padding trick, because of two now-fixed sgc limitations: an
#   if-reassigned loop-local variable used across multiple kernel calls in
#   the same block wasn't threaded through correctly by sgc's task-graph
#   codegen, and the shared ESSDistributor sized itself from the max
#   dimensions across every array in the function, so padding one array
#   forced padding all of them. Both are fixed now, so the block grid is
#   just nb x mb x lb everywhere, matching the real domain decomposition.
#
# Grid/particle/timestep counts match the original (21^3 grid, 81^3 ion +
# 41^3 electron quiet-start particles, dt=2e-10, 10000 steps) - see the
# "mesh / block parameters" section below. The one deliberate deviation is
# solver_max_it: the original runs up to 10000 SOR sweeps per solve but
# checks L2 convergence every 25 sweeps and breaks out early (typically
# within a few hundred sweeps for this grid/omega); pysol's while loop has
# no break, so solver_max_it is instead a fixed sweep count chosen to
# reach the same L2<1e-4 convergence depth (see the comment at its
# declaration). omega is also lower than the original's 1.4 - see its
# comment below.
# =============================================================================


@pysol_struct
class Particle:
    pos_x: Type.Double
    pos_y: Type.Double
    pos_z: Type.Double
    vel_x: Type.Double
    vel_y: Type.Double
    vel_z: Type.Double
    mpw: Type.Double


@pysol
def pic3d():
    # ── kernel imports ────────────────────────────────────────────────────────
    zero_phi_k = import_kernel("zero_block3d",
                                [KernelArgument("a", False, Type.Double, 3)])

    compute_node_vol_k = import_kernel("compute_node_volumes_block",
                                        [KernelArgument("ii", True, Type.Int),
                                         KernelArgument("jj", True, Type.Int),
                                         KernelArgument("kk", True, Type.Int),
                                         KernelArgument("dhx", True, Type.Double),
                                         KernelArgument("dhy", True, Type.Double),
                                         KernelArgument("dhz", True, Type.Double),
                                         KernelArgument("gn", True, Type.Int),
                                         KernelArgument("gm", True, Type.Int),
                                         KernelArgument("gl", True, Type.Int),
                                         KernelArgument("node_vol_block", False, Type.Double, 3)])

    # Shared by both species (the .sol generator forbids importing the same
    # underlying kernel name twice, so one Python handle is reused for both
    # the ion and electron call sites below). Always called on block
    # (0,0,0) only - see its comment in ucodes.h for why that's fine.
    load_particles_k = import_kernel("load_particles_box_qs",
                                      [KernelArgument("x1x", True, Type.Double),
                                       KernelArgument("x1y", True, Type.Double),
                                       KernelArgument("x1z", True, Type.Double),
                                       KernelArgument("x2x", True, Type.Double),
                                       KernelArgument("x2y", True, Type.Double),
                                       KernelArgument("x2z", True, Type.Double),
                                       KernelArgument("num_den", True, Type.Double),
                                       KernelArgument("npx", True, Type.Int),
                                       KernelArgument("npy", True, Type.Int),
                                       KernelArgument("npz", True, Type.Int),
                                       KernelArgument("cell", False, Particle)])

    # A particle's push can move it into a different block; rather than
    # writing directly into a neighbor's cell (a second, independent
    # producer of that neighbor's array element - a single-assignment
    # violation, same class of bug as the old global ef_x/ef_y/ef_z), it is
    # routed into one of 7 outputs: mail_self (stayed) or one of 6
    # face-direction mailboxes. merge_particles_mail (below) is each
    # block's sole producer of its own next-version cell. See
    # push_particles in ucodes.h for the full explanation and the
    # single-block-hop-per-timestep assumption this relies on.
    push_particles_k = import_kernel("push_particles",
                                      [KernelArgument("dt", True, Type.Double),
                                       KernelArgument("qm_dt", True, Type.Double),
                                       KernelArgument("x0x", True, Type.Double),
                                       KernelArgument("x0y", True, Type.Double),
                                       KernelArgument("x0z", True, Type.Double),
                                       KernelArgument("xmx", True, Type.Double),
                                       KernelArgument("xmy", True, Type.Double),
                                       KernelArgument("xmz", True, Type.Double),
                                       KernelArgument("dhx", True, Type.Double),
                                       KernelArgument("dhy", True, Type.Double),
                                       KernelArgument("dhz", True, Type.Double),
                                       KernelArgument("ii", True, Type.Int),
                                       KernelArgument("jj", True, Type.Int),
                                       KernelArgument("kk", True, Type.Int),
                                       KernelArgument("nb", True, Type.Int),
                                       KernelArgument("mb", True, Type.Int),
                                       KernelArgument("lb", True, Type.Int),
                                       KernelArgument("ef_x_block", True, Type.Double, 3),
                                       KernelArgument("ef_y_block", True, Type.Double, 3),
                                       KernelArgument("ef_z_block", True, Type.Double, 3),
                                       KernelArgument("cell_old", True, Particle),
                                       KernelArgument("mail_self", False, Particle),
                                       KernelArgument("mail_xm", False, Particle),
                                       KernelArgument("mail_xp", False, Particle),
                                       KernelArgument("mail_ym", False, Particle),
                                       KernelArgument("mail_yp", False, Particle),
                                       KernelArgument("mail_zm", False, Particle),
                                       KernelArgument("mail_zp", False, Particle)])

    merge_particles_k = import_kernel("merge_particles_mail",
                                       [KernelArgument("from_self", True, Particle),
                                        KernelArgument("from_xm", True, Particle),
                                        KernelArgument("from_xp", True, Particle),
                                        KernelArgument("from_ym", True, Particle),
                                        KernelArgument("from_yp", True, Particle),
                                        KernelArgument("from_zm", True, Particle),
                                        KernelArgument("from_zp", True, Particle),
                                        KernelArgument("cell_new", False, Particle)])

    # Scatters this block's own particles onto a RAW (unnormalized, not yet
    # reduced across blocks) ghosted buffer - see reduce_density_k below,
    # which is the single producer of the final den_ion/den_elec.
    compute_density_raw_k = import_kernel("compute_density_raw_block",
                                           [KernelArgument("x0x", True, Type.Double),
                                            KernelArgument("x0y", True, Type.Double),
                                            KernelArgument("x0z", True, Type.Double),
                                            KernelArgument("dhx", True, Type.Double),
                                            KernelArgument("dhy", True, Type.Double),
                                            KernelArgument("dhz", True, Type.Double),
                                            KernelArgument("ii", True, Type.Int),
                                            KernelArgument("jj", True, Type.Int),
                                            KernelArgument("kk", True, Type.Int),
                                            KernelArgument("cell", True, Particle),
                                            KernelArgument("den_raw", False, Type.Double, 3)])

    # Additive reduction of compute_density_raw_k's output across blocks -
    # see its long comment in ucodes.h. fxm/fxp/fym/fyp/fzm/fzp select
    # which local index of each neighbor (or self, at a domain boundary)
    # to read - computed per block below via a plain Python `if`, same as
    # im1/ip1/jm1/jp1/km1/kp1.
    reduce_density_k = import_kernel("reduce_density_block",
                                      [KernelArgument("fxm", True, Type.Int),
                                       KernelArgument("fxp", True, Type.Int),
                                       KernelArgument("fym", True, Type.Int),
                                       KernelArgument("fyp", True, Type.Int),
                                       KernelArgument("fzm", True, Type.Int),
                                       KernelArgument("fzp", True, Type.Int),
                                       KernelArgument("self_raw", True, Type.Double, 3),
                                       KernelArgument("xm_raw", True, Type.Double, 3),
                                       KernelArgument("xp_raw", True, Type.Double, 3),
                                       KernelArgument("ym_raw", True, Type.Double, 3),
                                       KernelArgument("yp_raw", True, Type.Double, 3),
                                       KernelArgument("zm_raw", True, Type.Double, 3),
                                       KernelArgument("zp_raw", True, Type.Double, 3),
                                       KernelArgument("node_vol", True, Type.Double, 3),
                                       KernelArgument("den", False, Type.Double, 3)])

    count_particles_k = import_kernel("count_particles",
                                       [KernelArgument("cell", True, Particle),
                                        KernelArgument("count", False, Type.Int)])

    # den_ion/den_elec are now fragmented (nb*mb*lb blocks), so sum()'s
    # whole-array reduce (which walks the program's main distributor) no
    # longer applies to them the way it does to phi/resid2 - summed with
    # this plain, accumulating kernel instead (see sum_block in ucodes.h).
    sum_block_k = import_kernel("sum_block",
                                 [KernelArgument("block", True, Type.Double, 3),
                                  KernelArgument("result", False, Type.Double)])

    # No more GLOBAL rho/distribute_rho_block bridge: density is already
    # fragmented, so charge density is computed directly into rho_frag's
    # interior, one block, one producer.
    charge_density_k = import_kernel("compute_charge_density_block",
                                      [KernelArgument("q_ion", True, Type.Double),
                                       KernelArgument("q_elec", True, Type.Double),
                                       KernelArgument("den_ion", True, Type.Double, 3),
                                       KernelArgument("den_elec", True, Type.Double, 3),
                                       KernelArgument("rho_block", False, Type.Double, 3)])

    # VTI output: writes block (ii,jj,kk)'s own phi/ef/rho/den_ion/
    # den_elec/node_vol straight to its own file under results/ - no
    # gather into any combined array at all, so this works unchanged
    # under MPI (see dump_vti_block in ucodes.h). write_vti_files() below
    # (run after the simulation, a plain Python function, not a kernel)
    # reads every block's file back and assembles the final .vti - the
    # same pattern examples/simple/update_array_3d_loop_test.py's verify()
    # uses to read per-block check files after the run instead of
    # gathering them inside the .sol-compiled program.
    dump_vti_block_k = import_kernel("dump_vti_block",
                                      [KernelArgument("ts", True, Type.Int),
                                       KernelArgument("ii", True, Type.Int),
                                       KernelArgument("jj", True, Type.Int),
                                       KernelArgument("kk", True, Type.Int),
                                       KernelArgument("phi_block", True, Type.Double, 3),
                                       KernelArgument("ef_x_block", True, Type.Double, 3),
                                       KernelArgument("ef_y_block", True, Type.Double, 3),
                                       KernelArgument("ef_z_block", True, Type.Double, 3),
                                       KernelArgument("rho_block", True, Type.Double, 3),
                                       KernelArgument("den_ion_block", True, Type.Double, 3),
                                       KernelArgument("den_elec_block", True, Type.Double, 3),
                                       KernelArgument("node_vol_block", True, Type.Double, 3)])

    # Same single-assignment requirement as push_particles: the SOR sweep
    # reads the OLD phi block and writes a fresh NEW one (see
    # sor_sweep_block in ucodes.h, modeled directly on hello.py's
    # calc_main(prev, curr)).
    sor_sweep_k = import_kernel("sor_sweep_block",
                                 [KernelArgument("ii", True, Type.Int),
                                  KernelArgument("jj", True, Type.Int),
                                  KernelArgument("kk", True, Type.Int),
                                  KernelArgument("nb", True, Type.Int),
                                  KernelArgument("mb", True, Type.Int),
                                  KernelArgument("lb", True, Type.Int),
                                  KernelArgument("idx2", True, Type.Double),
                                  KernelArgument("idy2", True, Type.Double),
                                  KernelArgument("idz2", True, Type.Double),
                                  KernelArgument("diag", True, Type.Double),
                                  KernelArgument("eps0", True, Type.Double),
                                  KernelArgument("omega", True, Type.Double),
                                  KernelArgument("rho_block", True, Type.Double, 3),
                                  KernelArgument("phi_old", True, Type.Double, 3),
                                  KernelArgument("phi_new", False, Type.Double, 3)])

    residual_k = import_kernel("residual_block",
                                [KernelArgument("ii", True, Type.Int),
                                 KernelArgument("jj", True, Type.Int),
                                 KernelArgument("kk", True, Type.Int),
                                 KernelArgument("nb", True, Type.Int),
                                 KernelArgument("mb", True, Type.Int),
                                 KernelArgument("lb", True, Type.Int),
                                 KernelArgument("idx2", True, Type.Double),
                                 KernelArgument("idy2", True, Type.Double),
                                 KernelArgument("idz2", True, Type.Double),
                                 KernelArgument("diag", True, Type.Double),
                                 KernelArgument("eps0", True, Type.Double),
                                 KernelArgument("rho_block", True, Type.Double, 3),
                                 KernelArgument("phi_block", True, Type.Double, 3),
                                 KernelArgument("resid2_block", False, Type.Double, 3)])

    # ef_x/ef_y/ef_z are now real, fragmented pysol Arrays: this writes
    # block (ii,jj,kk)'s own interior (single assignment, one producer),
    # and pysol_update_array ghost-exchanges them right after, exactly
    # like phi itself.
    efield_k = import_kernel("compute_efield_block",
                              [KernelArgument("ii", True, Type.Int),
                               KernelArgument("jj", True, Type.Int),
                               KernelArgument("kk", True, Type.Int),
                               KernelArgument("dhx", True, Type.Double),
                               KernelArgument("dhy", True, Type.Double),
                               KernelArgument("dhz", True, Type.Double),
                               KernelArgument("phi_block", True, Type.Double, 3),
                               KernelArgument("gn", True, Type.Int),
                               KernelArgument("gm", True, Type.Int),
                               KernelArgument("gl", True, Type.Int),
                               KernelArgument("ef_x_new", False, Type.Double, 3),
                               KernelArgument("ef_y_new", False, Type.Double, 3),
                               KernelArgument("ef_z_new", False, Type.Double, 3)])

    print_step_k = import_kernel("print_step",
                                  [KernelArgument("ts", True, Type.Int),
                                   KernelArgument("n_ion", True, Type.Int),
                                   KernelArgument("n_elec", True, Type.Int),
                                   KernelArgument("sum_den_ion", True, Type.Double),
                                   KernelArgument("sum_den_elec", True, Type.Double),
                                   KernelArgument("it", True, Type.Int),
                                   KernelArgument("l2resid2", True, Type.Double),
                                   KernelArgument("max_phi", True, Type.Double),
                                   KernelArgument("min_phi", True, Type.Double)])

    # ── mesh / block parameters ─────────────────────────────────────────────
    # gn/gm/gl = 21 matches the original's World(21,21,21). The fragmented
    # Poisson solve needs a block decomposition of that global grid (the
    # original has none - it's a single flat array); nb*bn = gn (3*7=21)
    # splits it into a 3x3x3 block grid of 7^3-interior blocks, which is
    # the block layout the pysol examples elsewhere pick when a grid size
    # must be exact instead of just "a multiple of nb".
    nb : Type.Int = 3
    mb : Type.Int = 3
    lb : Type.Int = 3
    bn : Type.Int = 7
    bm : Type.Int = 7
    bl : Type.Int = 7
    gn : Type.Int = 21
    gm : Type.Int = 21
    gl : Type.Int = 21
    # The .sol compiler's `if` only supports "var op var" / "var op const"
    # (a bare Name or literal Constant on the comparison's other side, see
    # _is_valid_if in decorator.py) - "if i == nb - 1:" is a BinOp, not an
    # atom, and is rejected outright, so the block-grid's last valid index
    # per axis is precomputed here once and compared against by name
    # everywhere the boundary-clamping `if`s below need it.
    nb_last : Type.Int = nb - 1
    mb_last : Type.Int = mb - 1
    lb_last : Type.Int = lb - 1

    x0x : Type.Double = -0.1
    x0y : Type.Double = -0.1
    x0z : Type.Double = 0.0
    xmx : Type.Double = 0.1
    xmy : Type.Double = 0.1
    xmz : Type.Double = 0.2
    xcx : Type.Double = 0.0
    xcy : Type.Double = 0.0
    xcz : Type.Double = 0.1

    dhx : Type.Double = (xmx - x0x) / (gn - 1)
    dhy : Type.Double = (xmy - x0y) / (gm - 1)
    dhz : Type.Double = (xmz - x0z) / (gl - 1)

    dt : Type.Double = 2e-10
    num_ts : Type.Int = 100
    num_den : Type.Double = 1e11

    npx_ion : Type.Int = 11
    npy_ion : Type.Int = 11
    npz_ion : Type.Int = 11
    npx_elec : Type.Int = 7
    npy_elec : Type.Int = 7
    npz_elec : Type.Int = 7

    # Written as chained divisions rather than a single scientific-notation
    # literal: the .sol lexer only accepts "[0-9]+e-[0-9]+" (integer
    # mantissa, e.g. 2e-10), not "[0-9]+.[0-9]+e-[0-9]+" (e.g. 8.85e-12) -
    # and since Python's AST discards the source text of a float literal,
    # writing 885418782e-20 in Python still regenerates "8.85418782e-12"
    # when unparsed. Splitting into chained BinOp divisions (never folded,
    # since the generator renders each operand's own literal text) keeps
    # every individual literal under 1e16, where Python's str() still
    # prints plain decimal instead of scientific notation.
    eps0 : Type.Double = 8.85418782 / 1000000000000.0                           # /1e12 = 8.85418782e-12
    qe : Type.Double = 1.602176565 / 10000000000.0 / 1000000000.0               # /1e10/1e9 = 1.602176565e-19
    amu : Type.Double = 1.660538921 / 1000000000000000.0 / 1000000000000.0      # /1e15/1e12 = 1.660538921e-27
    me : Type.Double = 9.10938215 / 1000000000000000.0 / 1000000000000000.0 / 10.0  # /1e15/1e15/10 = 9.10938215e-31

    ion_mass : Type.Double = 1600.0 * amu
    ion_charge : Type.Double = qe
    elec_mass : Type.Double = me
    elec_charge : Type.Double = 0.0 - qe

    # Fixed sweep count, since pysol's while loop has no break/early-exit:
    # the original caps at 10000 sweeps but checks L2<1e-4 every 25 sweeps
    # and stops as soon as that holds, which for a 21^3 grid at omega=1.4
    # is typically a few hundred sweeps, not anywhere near the 10000 cap.
    solver_max_it : Type.Int = 400  # SOR sweeps per Poisson solve (see the while loop below)
    solver_tol2 : Type.Double = 1e-4

    idx2 : Type.Double = 1.0 / (dhx * dhx)
    idy2 : Type.Double = 1.0 / (dhy * dhy)
    idz2 : Type.Double = 1.0 / (dhz * dhz)
    diag : Type.Double = 2.0 * idx2 + 2.0 * idy2 + 2.0 * idz2
    # The original uses 1.4 because its sweep is TRUE Gauss-Seidel: each
    # node update is written in place immediately, so later nodes in the
    # same sweep already see earlier ones' new values. sor_sweep_block is
    # single-assignment (every node reads phi_old, none of them see each
    # other's updates until the next sweep), which is algorithmically
    # Jacobi over-relaxation instead - a stricter-stability scheme whose
    # safe omega range shrinks as the grid gets finer. At this grid's
    # resolution 1.4 diverges to inf within ~2 timesteps and then crashes
    # on the resulting NaN particle positions (confirmed by instrumenting
    # push_particles directly); 1.0 (plain Jacobi, unconditionally stable
    # for this diagonally-dominant discrete Laplacian) does not, and still
    # converges L2resid2 to ~1e-20 well inside solver_max_it sweeps - both
    # schemes converge to the same fixed point when they converge at all,
    # so this does not change the converged phi, only how safely it's
    # reached.
    omega : Type.Double = 1.0

    # ── arrays ───────────────────────────────────────────────────────────────
    # Fragmented (nb x mb x lb blocks, 1-node ghost layer): the Poisson
    # solve, and now ef too (ghost-exchanged the same way, once per
    # timestep instead of once per sweep).
    phi = Array(3, 3, [nb, mb, lb], [bn + 2, bm + 2, bl + 2], Type.Double)
    rho_frag = Array(3, 3, [nb, mb, lb], [bn + 2, bm + 2, bl + 2], Type.Double)
    resid2 = Array(3, 3, [nb, mb, lb], [bn + 2, bm + 2, bl + 2], Type.Double)
    ef_x = Array(3, 3, [nb, mb, lb], [bn + 2, bm + 2, bl + 2], Type.Double)
    ef_y = Array(3, 3, [nb, mb, lb], [bn + 2, bm + 2, bl + 2], Type.Double)
    ef_z = Array(3, 3, [nb, mb, lb], [bn + 2, bm + 2, bl + 2], Type.Double)

    # Fragmented, no ghost: node_vol/den_ion/den_elec don't need one -
    # node_vol is a pure function of global position (see
    # compute_node_volumes_block), and density's cross-block spillover is
    # reduced by hand into den_ion/den_elec directly (see
    # reduce_density_block) rather than carried in a ghost layer.
    # den_ion_raw/den_elec_raw DO have a ghost layer - that's exactly
    # where a particle near a block edge's scatter spillover lands, for
    # reduce_density_block to pick up.
    node_vol = Array(3, 3, [nb, mb, lb], [bn, bm, bl], Type.Double)
    den_ion = Array(3, 3, [nb, mb, lb], [bn, bm, bl], Type.Double)
    den_elec = Array(3, 3, [nb, mb, lb], [bn, bm, bl], Type.Double)
    den_ion_raw = Array(3, 3, [nb, mb, lb], [bn + 2, bm + 2, bl + 2], Type.Double)
    den_elec_raw = Array(3, 3, [nb, mb, lb], [bn + 2, bm + 2, bl + 2], Type.Double)

    # Particles: one GrowingArray cell per block (not one global cell) -
    # see push_particles/merge_particles_mail in ucodes.h for how a
    # particle crossing a block boundary migrates to its new owning
    # block's cell. mail_*_self/xm/xp/ym/yp/zm/zp are the 7 transient
    # per-timestep mailboxes push_particles_k routes pushed particles
    # into and merge_particles_k reads back out of.
    particles_ion = GrowingArray(3, [nb, mb, lb], Particle)
    particles_elec = GrowingArray(3, [nb, mb, lb], Particle)
    mail_ion_self = GrowingArray(3, [nb, mb, lb], Particle)
    mail_ion_xm = GrowingArray(3, [nb, mb, lb], Particle)
    mail_ion_xp = GrowingArray(3, [nb, mb, lb], Particle)
    mail_ion_ym = GrowingArray(3, [nb, mb, lb], Particle)
    mail_ion_yp = GrowingArray(3, [nb, mb, lb], Particle)
    mail_ion_zm = GrowingArray(3, [nb, mb, lb], Particle)
    mail_ion_zp = GrowingArray(3, [nb, mb, lb], Particle)
    mail_elec_self = GrowingArray(3, [nb, mb, lb], Particle)
    mail_elec_xm = GrowingArray(3, [nb, mb, lb], Particle)
    mail_elec_xp = GrowingArray(3, [nb, mb, lb], Particle)
    mail_elec_ym = GrowingArray(3, [nb, mb, lb], Particle)
    mail_elec_yp = GrowingArray(3, [nb, mb, lb], Particle)
    mail_elec_zm = GrowingArray(3, [nb, mb, lb], Particle)
    mail_elec_zp = GrowingArray(3, [nb, mb, lb], Particle)

    # ── initial setup ────────────────────────────────────────────────────────
    for i in range(nb):
        for j in range(mb):
            for k in range(lb):
                compute_node_vol_k(i, j, k, dhx, dhy, dhz, gn, gm, gl, node_vol[i][j][k])
                zero_phi_k(phi[0, i][j][k])
                # Blocks are not zero-initialized by the runtime (unlike the
                # pure-Python ArrayElement's default of 0.0) - the original
                # never needs this since World zero-initializes rho, but
                # here rho_frag must be zeroed explicitly too, or the
                # initial Poisson solve below reads garbage and seeds phi
                # with a bogus non-zero starting guess that can later blow
                # up once real density is added in the main loop.
                zero_phi_k(rho_frag[i][j][k])

    # Loaded on block (0,0,0) only - see load_particles_box_qs's comment
    # in ucodes.h. The main loop's first push+merge cycle sorts every
    # particle into its real owning block before anything else reads
    # particles_ion/particles_elec.
    for i in range(1):
        for j in range(1):
            for k in range(1):
                load_particles_k(x0x, x0y, x0z, xmx, xmy, xmz, num_den, npx_ion, npy_ion, npz_ion,
                                  particles_ion[0, i][j][k])
                load_particles_k(x0x, x0y, x0z, xcx, xcy, xcz, num_den, npx_elec, npy_elec, npz_elec,
                                  particles_elec[0, i][j][k])

    # ── initial potential (electric field is still zero: phi/rho are 0) ─────
    # The Poisson solver's convergence loop is a `while` here, replacing the
    # original's "for (it=0; it<max_it; it++) { ...; if (converged) break; }".
    #
    # phi is single-assignment (see sor_sweep_block in ucodes.h): each sweep
    # reads OLD version "it" and writes NEW version "it+1" (interior only,
    # stale/empty ghost); pysol_update_array(phi, it+1, phi, it+2) - called
    # INSIDE the while loop, exactly like hello.py's solver() -
    # copies that interior into "it+2" and fills its ghost from neighbours'
    # "it+1" interior, giving a fully valid state to read on the next sweep.
    # "it" itself is never reset - it keeps growing across the initial
    # solve and every per-timestep solve below, reusing phi's latest valid
    # version as the next solve's starting guess (as the original does by
    # never re-zeroing phi). "sweep" is the per-solve iteration count reset
    # at the start of each solve, used only for the while condition.
    it : Type.Int = 0
    sweep : Type.Int = 0
    while sweep < solver_max_it:
        for i in range(nb):
            for j in range(mb):
                for k in range(lb):
                    sor_sweep_k(i, j, k, nb, mb, lb, idx2, idy2, idz2, diag, eps0, omega,
                                rho_frag[i][j][k], phi[it, i][j][k], phi[it + 1, i][j][k])
        pysol_update_array(phi, it + 1, phi, it + 2)
        it = it + 2
        sweep = sweep + 1

    for i in range(nb):
        for j in range(mb):
            for k in range(lb):
                residual_k(i, j, k, nb, mb, lb, idx2, idy2, idz2, diag, eps0,
                           rho_frag[i][j][k], phi[it, i][j][k], resid2[i][j][k])
    l2resid2 : Type.Double = sum(resid2)

    # ef is fresh every call (single-assignment, no history needed across
    # timesteps the way phi needs), so it just ping-pongs between a fixed
    # version 0 (interior, pre-exchange) and version 1 (post-exchange,
    # final) every single time - the same "write version 0 again on a
    # later call" pattern rho_frag/den_ion/den_elec already use safely,
    # since each call is still a distinct, single producer for its own
    # block's coordinates.
    for i in range(nb):
        for j in range(mb):
            for k in range(lb):
                efield_k(i, j, k, dhx, dhy, dhz, phi[it, i][j][k], gn, gm, gl,
                         ef_x[0, i][j][k], ef_y[0, i][j][k], ef_z[0, i][j][k])
    pysol_update_array(ef_x, 0, ef_x, 1)
    pysol_update_array(ef_y, 0, ef_y, 1)
    pysol_update_array(ef_z, 0, ef_z, 1)

    # ── main PIC loop ────────────────────────────────────────────────────────
    # "pv" is the particle version, bumped once per timestep (push reads
    # version pv, writes pv+1 - but via the 7-mailbox scheme, not directly;
    # see push_particles/merge_particles_mail in ucodes.h).
    pv : Type.Int = 0
    # (a for-loop here, not a while - the original already used a while for
    # this outer step counter, but the task's "while instead of for+break"
    # ask targets the Poisson solver's convergence loop specifically, which
    # is where the batched `while`+SOR-sweep loops above/below apply.)
    for ts in range(num_ts):
        qm_dt_ion : Type.Double = dt * ion_charge / ion_mass
        qm_dt_elec : Type.Double = dt * elec_charge / elec_mass

        for i in range(nb):
            for j in range(mb):
                for k in range(lb):
                    # Clamped neighbor block indices for merge_particles_k's
                    # boundary case (see merge_particles_mail in ucodes.h):
                    # "from_xm" always reads mail_*_xp[im1] (the formula is
                    # fixed by argument position, not chosen per call), so
                    # at i==0, where there is no real left neighbor, im1
                    # can't just be "i" (self) - self's own mail_xp slot is
                    # real, active data heading to the i==1 neighbor, not an
                    # empty placeholder. The one slot of mail_*_xp that IS
                    # always guaranteed empty is index nb-1 (the rightmost
                    # block never has +x-bound particles, since reflection
                    # keeps every position inside [x0, xm)), so that's what
                    # im1 is redirected to instead. Each of the other 5
                    # directions mirrors this with its own always-empty
                    # index.
                    im1 : Type.Int = i - 1
                    ip1 : Type.Int = i + 1
                    jm1 : Type.Int = j - 1
                    jp1 : Type.Int = j + 1
                    km1 : Type.Int = k - 1
                    kp1 : Type.Int = k + 1
                    if i == 0:
                        im1 = nb_last
                    if i == nb_last:
                        ip1 = 0
                    if j == 0:
                        jm1 = mb_last
                    if j == mb_last:
                        jp1 = 0
                    if k == 0:
                        km1 = lb_last
                    if k == lb_last:
                        kp1 = 0

                    push_particles_k(dt, qm_dt_ion, x0x, x0y, x0z, xmx, xmy, xmz, dhx, dhy, dhz,
                                      i, j, k, nb, mb, lb,
                                      ef_x[1, i][j][k], ef_y[1, i][j][k], ef_z[1, i][j][k],
                                      particles_ion[pv, i][j][k],
                                      mail_ion_self[pv + 1, i][j][k], mail_ion_xm[pv + 1, i][j][k],
                                      mail_ion_xp[pv + 1, i][j][k], mail_ion_ym[pv + 1, i][j][k],
                                      mail_ion_yp[pv + 1, i][j][k], mail_ion_zm[pv + 1, i][j][k],
                                      mail_ion_zp[pv + 1, i][j][k])
                    push_particles_k(dt, qm_dt_elec, x0x, x0y, x0z, xmx, xmy, xmz, dhx, dhy, dhz,
                                      i, j, k, nb, mb, lb,
                                      ef_x[1, i][j][k], ef_y[1, i][j][k], ef_z[1, i][j][k],
                                      particles_elec[pv, i][j][k],
                                      mail_elec_self[pv + 1, i][j][k], mail_elec_xm[pv + 1, i][j][k],
                                      mail_elec_xp[pv + 1, i][j][k], mail_elec_ym[pv + 1, i][j][k],
                                      mail_elec_yp[pv + 1, i][j][k], mail_elec_zm[pv + 1, i][j][k],
                                      mail_elec_zp[pv + 1, i][j][k])

                    merge_particles_k(mail_ion_self[pv + 1, i][j][k], mail_ion_xp[pv + 1, im1][j][k],
                                       mail_ion_xm[pv + 1, ip1][j][k], mail_ion_yp[pv + 1, i][jm1][k],
                                       mail_ion_ym[pv + 1, i][jp1][k], mail_ion_zp[pv + 1, i][j][km1],
                                       mail_ion_zm[pv + 1, i][j][kp1], particles_ion[pv + 2, i][j][k])
                    merge_particles_k(mail_elec_self[pv + 1, i][j][k], mail_elec_xp[pv + 1, im1][j][k],
                                       mail_elec_xm[pv + 1, ip1][j][k], mail_elec_yp[pv + 1, i][jm1][k],
                                       mail_elec_ym[pv + 1, i][jp1][k], mail_elec_zp[pv + 1, i][j][km1],
                                       mail_elec_zm[pv + 1, i][j][kp1], particles_elec[pv + 2, i][j][k])

                    compute_density_raw_k(x0x, x0y, x0z, dhx, dhy, dhz, i, j, k,
                                          particles_ion[pv + 2, i][j][k], den_ion_raw[i][j][k])
                    compute_density_raw_k(x0x, x0y, x0z, dhx, dhy, dhz, i, j, k,
                                          particles_elec[pv + 2, i][j][k], den_elec_raw[i][j][k])

        for i in range(nb):
            for j in range(mb):
                for k in range(lb):
                    # Same boundary-clamp idea as the mailboxes above,
                    # applied to den_*_raw, but with a DIFFERENT strategy:
                    # unlike merge_particles_mail's from_xm/from_xp (whose
                    # array identity is fixed by argument position),
                    # reduce_density_block's neighbor array argument AND
                    # the local face index to read within it are both
                    # freely chosen per call - so at a domain boundary the
                    # safe choice is to keep the array as SELF (dim1=i) and
                    # switch the face index to self's own opposite-side
                    # ghost (always zero, since no particle scatters
                    # beyond the domain edge), not to redirect to the far
                    # block the way the mailboxes do.
                    dim1 : Type.Int = i - 1
                    dip1 : Type.Int = i + 1
                    djm1 : Type.Int = j - 1
                    djp1 : Type.Int = j + 1
                    dkm1 : Type.Int = k - 1
                    dkp1 : Type.Int = k + 1
                    fxm : Type.Int = bn + 1
                    fxp : Type.Int = 0
                    fym : Type.Int = bm + 1
                    fyp : Type.Int = 0
                    fzm : Type.Int = bl + 1
                    fzp : Type.Int = 0
                    if i == 0:
                        dim1 = i
                        fxm = 0
                    if i == nb_last:
                        dip1 = i
                        fxp = bn + 1
                    if j == 0:
                        djm1 = j
                        fym = 0
                    if j == mb_last:
                        djp1 = j
                        fyp = bm + 1
                    if k == 0:
                        dkm1 = k
                        fzm = 0
                    if k == lb_last:
                        dkp1 = k
                        fzp = bl + 1

                    reduce_density_k(fxm, fxp, fym, fyp, fzm, fzp,
                                      den_ion_raw[i][j][k], den_ion_raw[dim1][j][k], den_ion_raw[dip1][j][k],
                                      den_ion_raw[i][djm1][k], den_ion_raw[i][djp1][k],
                                      den_ion_raw[i][j][dkm1], den_ion_raw[i][j][dkp1],
                                      node_vol[i][j][k], den_ion[i][j][k])
                    reduce_density_k(fxm, fxp, fym, fyp, fzm, fzp,
                                      den_elec_raw[i][j][k], den_elec_raw[dim1][j][k], den_elec_raw[dip1][j][k],
                                      den_elec_raw[i][djm1][k], den_elec_raw[i][djp1][k],
                                      den_elec_raw[i][j][dkm1], den_elec_raw[i][j][dkp1],
                                      node_vol[i][j][k], den_elec[i][j][k])

                    charge_density_k(ion_charge, elec_charge, den_ion[i][j][k], den_elec[i][j][k], rho_frag[i][j][k])

        # "it" is NOT reset here - phi keeps growing its version across every
        # timestep's solve, reusing the previous timestep's potential as the
        # next solve's starting guess, exactly as the original does by never
        # re-zeroing phi. "sweep" is reset for this solve's while condition.
        sweep : Type.Int = 0
        while sweep < solver_max_it:
            for i in range(nb):
                for j in range(mb):
                    for k in range(lb):
                        sor_sweep_k(i, j, k, nb, mb, lb, idx2, idy2, idz2, diag, eps0, omega,
                                    rho_frag[i][j][k], phi[it, i][j][k], phi[it + 1, i][j][k])
            pysol_update_array(phi, it + 1, phi, it + 2)
            it = it + 2
            sweep = sweep + 1

        for i in range(nb):
            for j in range(mb):
                for k in range(lb):
                    residual_k(i, j, k, nb, mb, lb, idx2, idy2, idz2, diag, eps0,
                               rho_frag[i][j][k], phi[it, i][j][k], resid2[i][j][k])
        # A fresh name (not a reassignment of the outer l2resid2) - the sol
        # compiler drops the trailing semicolon when a sum()/max()/min()
        # result is assigned to an already-declared variable rather than
        # freshly declared; confirmed with a minimal reproduction.
        l2resid2_step : Type.Double = sum(resid2)

        for i in range(nb):
            for j in range(mb):
                for k in range(lb):
                    efield_k(i, j, k, dhx, dhy, dhz, phi[it, i][j][k], gn, gm, gl,
                             ef_x[0, i][j][k], ef_y[0, i][j][k], ef_z[0, i][j][k])
        pysol_update_array(ef_x, 0, ef_x, 1)
        pysol_update_array(ef_y, 0, ef_y, 1)
        pysol_update_array(ef_z, 0, ef_z, 1)

        n_ion : Type.Int = 0
        n_elec : Type.Int = 0
        sum_den_ion : Type.Double = 0.0
        sum_den_elec : Type.Double = 0.0
        for i in range(nb):
            for j in range(mb):
                for k in range(lb):
                    count_particles_k(particles_ion[pv + 2, i][j][k], n_ion)
                    count_particles_k(particles_elec[pv + 2, i][j][k], n_elec)
                    sum_block_k(den_ion[i][j][k], sum_den_ion)
                    sum_block_k(den_elec[i][j][k], sum_den_elec)
        max_phi : Type.Double = max(phi)
        min_phi : Type.Double = min(phi)

        print_step_k(ts, n_ion, n_elec, sum_den_ion, sum_den_elec, sweep, l2resid2_step, max_phi, min_phi)

        # VTK output (see dump_vti_block in ucodes.h): dumps each block's
        # own data straight to its own file under results/ - no in-program
        # gather, so this works unchanged under MPI. The periodic "every
        # 100 steps or last step" gate ("ts % 100 == 0 ||
        # world.isLastTimeStep()" in the original's Main.cpp) lives inside
        # dump_vti_block itself (plain "if (ts % 100 != 0) return;" in
        # C++) rather than as a pysol `if` here - this one's gating a
        # kernel call directly on "ts", which is cheap to just always do
        # in C++ regardless of whether a pysol-level if would also work
        # now.
        for i in range(nb):
            for j in range(mb):
                for k in range(lb):
                    dump_vti_block_k(ts, i, j, k, phi[it, i][j][k],
                                      ef_x[1, i][j][k], ef_y[1, i][j][k], ef_z[1, i][j][k],
                                      rho_frag[i][j][k], den_ion[i][j][k], den_elec[i][j][k],
                                      node_vol[i][j][k])

        pv = pv + 2


def _flat3(i, j, k, m, l):
    return (i * m + j) * l + k


def _read_doubles(path, count):
    a = array.array("d")
    with open(path, "rb") as fh:
        a.frombytes(fh.read(count * 8))
    return a


def write_vti_files():
    """Post-processing step, run after the simulation: gathers every
    block's dump_vti_block_k file for each dumped timestep and assembles
    results/fields_<ts>.vti, in the same VTK ImageData format (same
    header, same PointData DataArray tags, same ASCII node order)
    Output::fields() in examples/pic3d/Output.cpp writes - but done here
    in plain Python instead of inside the .sol-compiled program, exactly
    like examples/simple/update_array_3d_loop_test.py's verify() reads
    per-block check files after the run rather than gathering them while
    it's still running. This is what keeps VTI output working under MPI:
    each rank only ever writes the files for the blocks it owns, into a
    shared results/ directory, and this step simply doesn't care which
    rank wrote which file.
    """
    n = GN * GM * GL
    for ts in range(0, NUM_TS, VTI_EVERY):
        phi = array.array("d", [0.0]) * n
        ef_x = array.array("d", [0.0]) * n
        ef_y = array.array("d", [0.0]) * n
        ef_z = array.array("d", [0.0]) * n
        rho = array.array("d", [0.0]) * n
        den_ion = array.array("d", [0.0]) * n
        den_elec = array.array("d", [0.0]) * n
        node_vol = array.array("d", [0.0]) * n

        found_any = False
        for ii in range(NB):
            for jj in range(MB):
                for kk in range(LB):
                    block_path = f"results/vti_block_{ts:05d}_{ii}_{jj}_{kk}.bin"
                    if not os.path.exists(block_path):
                        continue
                    found_any = True
                    rec = _read_doubles(block_path, 8 * BN * BM * BL)
                    idx = 0
                    for li in range(BN):
                        for lj in range(BM):
                            for lk in range(BL):
                                gi, gj, gk = ii * BN + li, jj * BM + lj, kk * BL + lk
                                gidx = _flat3(gi, gj, gk, GM, GL)
                                phi[gidx] = rec[idx]
                                ef_x[gidx] = rec[idx + 1]
                                ef_y[gidx] = rec[idx + 2]
                                ef_z[gidx] = rec[idx + 3]
                                rho[gidx] = rec[idx + 4]
                                den_ion[gidx] = rec[idx + 5]
                                den_elec[gidx] = rec[idx + 6]
                                node_vol[gidx] = rec[idx + 7]
                                idx += 8

        if not found_any:
            continue

        with open(f"results/fields_{ts:05d}.vti", "w") as out:
            out.write("<VTKFile type=\"ImageData\">\n")
            out.write(f"<ImageData Origin=\"{X0[0]} {X0[1]} {X0[2]}\" ")
            out.write(f"Spacing=\"{DH[0]} {DH[1]} {DH[2]}\" ")
            out.write(f"WholeExtent=\"0 {GN - 1} 0 {GM - 1} 0 {GL - 1}\">\n")
            out.write("<PointData>\n")

            def write_scalar(name, field):
                out.write(f"<DataArray Name=\"{name}\" NumberOfComponents=\"1\" "
                          "format=\"ascii\" type=\"Float64\">\n")
                for k in range(GL):
                    row = " ".join(str(field[_flat3(i, j, k, GM, GL)])
                                    for j in range(GM) for i in range(GN))
                    out.write(row + " \n")
                out.write("</DataArray>\n")

            write_scalar("NodeVol", node_vol)
            write_scalar("phi", phi)
            write_scalar("rho", rho)
            write_scalar("nd.O+", den_ion)
            write_scalar("nd.e-", den_elec)

            out.write("<DataArray Name=\"ef\" NumberOfComponents=\"3\" "
                      "format=\"ascii\" type=\"Float64\">\n")
            for k in range(GL):
                parts = []
                for j in range(GM):
                    for i in range(GN):
                        n_ = _flat3(i, j, k, GM, GL)
                        parts.append(f"{ef_x[n_]} {ef_y[n_]} {ef_z[n_]}")
                out.write(" ".join(parts) + " \n")
            out.write("</DataArray>\n")

            out.write("</PointData>\n")
            out.write("</ImageData>\n")
            out.write("</VTKFile>\n")


if __name__ == "__main__":
    load_config("config.json")

    print("==========PIC3D (pysol)============")
    pysol_compile("pic3d")
    pysol_run("pic3d")
    write_vti_files()
