
import array
import os

from pysol import (
    Array, GrowingArray, KernelArgument, Type, import_kernel, pysol,
    pysol_compile, pysol_run, load_config, pysol_struct,
    pysol_update_array_3d_26_full_sum,
)

# Module-level mirror of pic3d() internal mesh/block parameters (the
# @pysol AST analysis only understands literals local to the decorated
# function, so write_density_files() below - plain Python, run after
# pysol_run() - needs its own copies to know the fragment layout).
NFX, NFY, NFZ = 10, 10, 10
INTERIOR = 40
BN, BM, BL = INTERIOR + 2, INTERIOR + 2, INTERIOR + 2
NUM_TS = 100
NTS_DUMP = 5
# Domain center/spacing mirrors (cx,cy,cz,hx,hy,hz above), needed to locate
# the center-plane slice index for the ro.cxy/cxz/cyz.NNNNNN files below.
CX, CY, CZ = 0.5, 0.5, 0.5
HX, HY, HZ = 1.0 / (NFX * INTERIOR), 1.0 / (NFY * INTERIOR), 1.0 / (NFZ * INTERIOR)

# =============================================================================
# pysol port of pic3d_common's self-gravitating N-body PIC (NOT
# electrostatic: Laplacian(fi) = 4*PI*rho with G=1, a stationary "sun"
# point mass sits at the domain center, particles reflect off the domain
# walls - see pic3d_common/common/{pic3d.h,mesh3d.h,pic3d.cpp} and
# pic3d_common/luna5/pic3d_luna5.{fa,cpp}, which this follows closely).
#
# EVERYTHING is fragmented into an nfx x nfy x nfz fragment grid, matching
# pic3d_common's own domain decomposition, with NO global, non-fragmented
# array anywhere - every cross-fragment dependency goes through a real
# pysol Array/GrowingArray read. Unlike pysol's OTHER pic3d port (examples/
# pic3d), which only needed 6-face ghost exchange, this reference genuinely
# needs the full 26-neighbor (face+edge+corner) pattern for BOTH its
# additive density merge and its copy potential-ghost refresh, plus a
# 27-direction (26+self) particle mailbox scheme for migration - see
# ucodes.h's module comment for the full exchange design (26
# "existsXYZ"-flagged neighbor arguments per exchange kernel call, every
# direction computed once per fragment from six has_lo_*/has_hi_* flags
# and reused for density, potential and particle exchanges alike).
#
# Deliberate simplifications relative to pic3d_common (all physics-
# adjacent, not exchange-adjacent - the exchange/fragmentation design
# itself is kept faithful):
#   - Initial particles: a deterministic lattice-in-disk (XY plane, within
#     `radius` of the center) with ZERO initial velocity, instead of the
#     reference's ring/bisection density-profile loader
#     (fill_particles_coords) and circular-orbit quiet start
#     (fill_particles_velocity). All particles are generated on fragment
#     (0,0,0); the first push+merge round sorts every one of them into
#     its real owning fragment (safe here since nfx=nfy=nfz=2, so no
#     particle can ever be more than one fragment-hop away in any axis).
#   - Diagnostics: particle count + total kinetic energy, not the
#     reference's full 13-field properties vector (mass center, momentum,
#     angular momentum, potential/kinetic energy split, sun potential
#     energy separately).
#   - The moving "center" (sun position) is passed as plain scalars
#     (cx,cy,cz), not a separate versioned array updated by move_center -
#     the reference's own default pic3d.par already has vx=vy=vz=0, so
#     the center never moves there either; this is a direct simplification
#     of that default, not a deviation from it.
#   - Jacobi sweep count is a fixed cap (max_jacobi_iters), like pysol's
#     other pic3d port's solver_max_it: pysol's for-loop can't break
#     early on a data-dependent convergence check, so this is a plain
#     generous fixed count instead of the reference's while(notstop).
#
# Known limitation, partially root-caused: there were actually TWO
# distinct bugs hiding behind the single symptom "n_particles/KE printed
# per step drifts under multithreading."
#
# (1) FIXED: count_particles_k/sum_kinetic_energy_k (now removed) wrote
# into one SHARED scalar via += across every one of the nfx*nfy*nfz
# kernel calls in a timestep, with no synchronization - a plain lost-
# update race confined to that diagnostic accumulator. Confirmed via
# independent per-fragment particle dumps cross-checked against the
# printed diagnostic (40+ repeated runs, up to 1000 fragments, ~282,000
# particles, 20 OMP threads): the actual particle data was exact and
# identical to the single-threaded baseline in every run, including ones
# where the OLD printed total was wrong - the simulation was never
# losing particles at that scale, only the diagnostic's own total was.
# See particle_diag_block in ucodes.h for the fix (each fragment writes
# its own count/KE into its own Array cell; the total is a real sgc
# sum() reduction, not a hand-rolled one).
#
# (2) NOT FIXED, confirmed real and distinct from (1): at the reference's
# actual scale (1,000,996 particles, bn=bm=bl=42, nfx=nfy=nfz=10), the
# same independent-dump method shows genuine particle-data corruption
# under multithreading (10 OMP threads) that (1)'s fix does not touch -
# e.g. one 3-timestep run showed 18 distinct (timestep, fragment) cells
# with particle counts AND proportionally-scaled KE differing from the
# 1-thread baseline (not just the count, which is what distinguishes
# this from bug (1)). Every single differing fragment was in the k=5
# layer - not a coincidence: every particle's initial z equals cx/cy/cz's
# cz=0.5 exactly (see generate_particles_disk_k), so the ENTIRE initial
# particle population starts in one single z-layer of fragments instead
# of being spread across all ten, a ~100x per-fragment load imbalance
# relative to every other z-layer at ts=0. This is almost certainly why
# it wasn't reproducible at smaller scales tested while chasing bug (1)
# (up to 282,000 particles, same skewed-IC shape but a proportionally
# much smaller absolute imbalance) - this looks like a genuine RTS-level
# race in GrowingArray cross-fragment read/write synchronization that
# only gets exposed under heavy, uneven per-fragment load, the same class
# of issue originally (and, it turns out, only partially correctly)
# blamed for bug (1). No reliable pic3d.py/ucodes.h-only workaround for
# this one was found - see push_particles_block/merge_particles_block in
# ucodes.h. Until it's fixed (most likely needs RTS-level work, out of
# this port's scope), run with OMP_NUM_THREADS=1 at the reference's full
# particle/fragment scale specifically; smaller configurations (as
# exercised by this repo's own tests) were not observed to be affected.
# =============================================================================


@pysol_struct
class Particle:
    x: Type.Double
    y: Type.Double
    z: Type.Double
    u: Type.Double
    v: Type.Double
    w: Type.Double


@pysol
def pic3d_pysol():

    # -- kernel imports --
    generate_particles_disk_k = import_kernel("generate_particles_disk_k",
                                              [KernelArgument("cx", True, Type.Double),
                                               KernelArgument("cy", True, Type.Double),
                                               KernelArgument("cz", True, Type.Double),
                                               KernelArgument("radius", True, Type.Double),
                                               KernelArgument("hx", True, Type.Double),
                                               KernelArgument("hy", True, Type.Double),
                                               KernelArgument("hz", True, Type.Double),
                                               KernelArgument("ii", True, Type.Int),
                                               KernelArgument("jj", True, Type.Int),
                                               KernelArgument("kk", True, Type.Int),
                                               KernelArgument("bn", True, Type.Int),
                                               KernelArgument("bm", True, Type.Int),
                                               KernelArgument("bl", True, Type.Int),
                                               KernelArgument("npx", True, Type.Int),
                                               KernelArgument("npy", True, Type.Int),
                                               KernelArgument("cell", False, Particle)])

    compute_density_raw_k = import_kernel("compute_density_raw_block",
                                          [KernelArgument("hx", True, Type.Double),
                                           KernelArgument("hy", True, Type.Double),
                                           KernelArgument("hz", True, Type.Double),
                                           KernelArgument("am", True, Type.Double),
                                           KernelArgument("ii", True, Type.Int),
                                           KernelArgument("jj", True, Type.Int),
                                           KernelArgument("kk", True, Type.Int),
                                           KernelArgument("bn", True, Type.Int),
                                           KernelArgument("bm", True, Type.Int),
                                           KernelArgument("bl", True, Type.Int),
                                           KernelArgument("cell", True, Particle),
                                           KernelArgument("den_raw", False, Type.Double, 3)])

    # Stage 1 ("read") of the potential exchange's two-stage ghost-exchange
    # split: copies a block's own data into a stable scratch array before
    # any neighbor reads it, the same way pysol_update_array's
    # _read_ghost3d_ kernel (see exchange_3d.sol) extracts a block's own
    # data before any other block's stage-2 write can run - ParSolGen
    # can't read and write the same array element within one kernel, so
    # the read (copy into the stable scratch) and the write (the actual
    # exchange, reading only from that now-settled scratch copy) must be
    # two separate kernel-call passes, not one combined pass. The density
    # exchange used to need this same split too, via its own
    # exchange_density_additive_block kernel - that is now the generic
    # pysol_update_array_3d_26_full_sum runtime primitive instead (see
    # exchange_3d_26_full_sum.sol), which does its own stage-1/stage-2
    # split internally.
    copy_block3d_k = import_kernel("copy_block3d",
                                    [KernelArgument("n", True, Type.Int),
                                     KernelArgument("m", True, Type.Int),
                                     KernelArgument("l", True, Type.Int),
                                     KernelArgument("a", True, Type.Double, 3),
                                     KernelArgument("b", False, Type.Double, 3)])

    exchange_potential_copy_k = import_kernel("exchange_potential_copy_block",
                                              [KernelArgument("self_fi", True, Type.Double, 3),
        KernelArgument("nbr000", True, Type.Double, 3),
        KernelArgument("exists000", True, Type.Int),
        KernelArgument("nbr001", True, Type.Double, 3),
        KernelArgument("exists001", True, Type.Int),
        KernelArgument("nbr002", True, Type.Double, 3),
        KernelArgument("exists002", True, Type.Int),
        KernelArgument("nbr010", True, Type.Double, 3),
        KernelArgument("exists010", True, Type.Int),
        KernelArgument("nbr011", True, Type.Double, 3),
        KernelArgument("exists011", True, Type.Int),
        KernelArgument("nbr012", True, Type.Double, 3),
        KernelArgument("exists012", True, Type.Int),
        KernelArgument("nbr020", True, Type.Double, 3),
        KernelArgument("exists020", True, Type.Int),
        KernelArgument("nbr021", True, Type.Double, 3),
        KernelArgument("exists021", True, Type.Int),
        KernelArgument("nbr022", True, Type.Double, 3),
        KernelArgument("exists022", True, Type.Int),
        KernelArgument("nbr100", True, Type.Double, 3),
        KernelArgument("exists100", True, Type.Int),
        KernelArgument("nbr101", True, Type.Double, 3),
        KernelArgument("exists101", True, Type.Int),
        KernelArgument("nbr102", True, Type.Double, 3),
        KernelArgument("exists102", True, Type.Int),
        KernelArgument("nbr110", True, Type.Double, 3),
        KernelArgument("exists110", True, Type.Int),
        KernelArgument("nbr112", True, Type.Double, 3),
        KernelArgument("exists112", True, Type.Int),
        KernelArgument("nbr120", True, Type.Double, 3),
        KernelArgument("exists120", True, Type.Int),
        KernelArgument("nbr121", True, Type.Double, 3),
        KernelArgument("exists121", True, Type.Int),
        KernelArgument("nbr122", True, Type.Double, 3),
        KernelArgument("exists122", True, Type.Int),
        KernelArgument("nbr200", True, Type.Double, 3),
        KernelArgument("exists200", True, Type.Int),
        KernelArgument("nbr201", True, Type.Double, 3),
        KernelArgument("exists201", True, Type.Int),
        KernelArgument("nbr202", True, Type.Double, 3),
        KernelArgument("exists202", True, Type.Int),
        KernelArgument("nbr210", True, Type.Double, 3),
        KernelArgument("exists210", True, Type.Int),
        KernelArgument("nbr211", True, Type.Double, 3),
        KernelArgument("exists211", True, Type.Int),
        KernelArgument("nbr212", True, Type.Double, 3),
        KernelArgument("exists212", True, Type.Int),
        KernelArgument("nbr220", True, Type.Double, 3),
        KernelArgument("exists220", True, Type.Int),
        KernelArgument("nbr221", True, Type.Double, 3),
        KernelArgument("exists221", True, Type.Int),
        KernelArgument("nbr222", True, Type.Double, 3),
        KernelArgument("exists222", True, Type.Int),
                                               KernelArgument("merged", False, Type.Double, 3)])

    init_boundary_potential_k = import_kernel("init_boundary_potential_block",
                                              [KernelArgument("has_lo_x", True, Type.Int),
                                               KernelArgument("has_hi_x", True, Type.Int),
                                               KernelArgument("has_lo_y", True, Type.Int),
                                               KernelArgument("has_hi_y", True, Type.Int),
                                               KernelArgument("has_lo_z", True, Type.Int),
                                               KernelArgument("has_hi_z", True, Type.Int),
                                               KernelArgument("hx", True, Type.Double),
                                               KernelArgument("hy", True, Type.Double),
                                               KernelArgument("hz", True, Type.Double),
                                               KernelArgument("cx", True, Type.Double),
                                               KernelArgument("cy", True, Type.Double),
                                               KernelArgument("cz", True, Type.Double),
                                               KernelArgument("pm", True, Type.Double),
                                               KernelArgument("ii", True, Type.Int),
                                               KernelArgument("jj", True, Type.Int),
                                               KernelArgument("kk", True, Type.Int),
                                               KernelArgument("bn", True, Type.Int),
                                               KernelArgument("bm", True, Type.Int),
                                               KernelArgument("bl", True, Type.Int),
                                               KernelArgument("fi", False, Type.Double, 3)])

    poisson_jacobi_sweep_k = import_kernel("poisson_jacobi_sweep_block",
                                           [KernelArgument("hx", True, Type.Double),
                                            KernelArgument("hy", True, Type.Double),
                                            KernelArgument("hz", True, Type.Double),
                                            KernelArgument("w", True, Type.Double),
                                            KernelArgument("ro", True, Type.Double, 3),
                                            KernelArgument("fi_old", True, Type.Double, 3),
                                            KernelArgument("fi_new", False, Type.Double, 3)])

    add_sun_potential_k = import_kernel("add_sun_potential_block",
                                        [KernelArgument("hx", True, Type.Double),
                                         KernelArgument("hy", True, Type.Double),
                                         KernelArgument("hz", True, Type.Double),
                                         KernelArgument("cx", True, Type.Double),
                                         KernelArgument("cy", True, Type.Double),
                                         KernelArgument("cz", True, Type.Double),
                                         KernelArgument("sm", True, Type.Double),
                                         KernelArgument("ii", True, Type.Int),
                                         KernelArgument("jj", True, Type.Int),
                                         KernelArgument("kk", True, Type.Int),
                                         KernelArgument("bn", True, Type.Int),
                                         KernelArgument("bm", True, Type.Int),
                                         KernelArgument("bl", True, Type.Int),
                                         KernelArgument("fp", True, Type.Double, 3),
                                         KernelArgument("fi", False, Type.Double, 3)])

    compute_force_k = import_kernel("compute_force_block",
                                    [KernelArgument("hx", True, Type.Double),
                                     KernelArgument("hy", True, Type.Double),
                                     KernelArgument("hz", True, Type.Double),
                                     KernelArgument("fi", True, Type.Double, 3),
                                     KernelArgument("fx", False, Type.Double, 3),
                                     KernelArgument("fy", False, Type.Double, 3),
                                     KernelArgument("fz", False, Type.Double, 3)])

    push_particles_k = import_kernel("push_particles_block",
                                     [KernelArgument("tau", True, Type.Double),
                                      KernelArgument("xn", True, Type.Double),
                                      KernelArgument("yn", True, Type.Double),
                                      KernelArgument("zn", True, Type.Double),
                                      KernelArgument("hx", True, Type.Double),
                                      KernelArgument("hy", True, Type.Double),
                                      KernelArgument("hz", True, Type.Double),
                                      KernelArgument("ii", True, Type.Int),
                                      KernelArgument("jj", True, Type.Int),
                                      KernelArgument("kk", True, Type.Int),
                                      KernelArgument("bn", True, Type.Int),
                                      KernelArgument("bm", True, Type.Int),
                                      KernelArgument("bl", True, Type.Int),
                                      KernelArgument("fx", True, Type.Double, 3),
                                      KernelArgument("fy", True, Type.Double, 3),
                                      KernelArgument("fz", True, Type.Double, 3),
                                      KernelArgument("cell_old", True, Particle),
                                      KernelArgument("mail_self", False, Particle),
        KernelArgument("mail_000", False, Particle),
        KernelArgument("mail_001", False, Particle),
        KernelArgument("mail_002", False, Particle),
        KernelArgument("mail_010", False, Particle),
        KernelArgument("mail_011", False, Particle),
        KernelArgument("mail_012", False, Particle),
        KernelArgument("mail_020", False, Particle),
        KernelArgument("mail_021", False, Particle),
        KernelArgument("mail_022", False, Particle),
        KernelArgument("mail_100", False, Particle),
        KernelArgument("mail_101", False, Particle),
        KernelArgument("mail_102", False, Particle),
        KernelArgument("mail_110", False, Particle),
        KernelArgument("mail_112", False, Particle),
        KernelArgument("mail_120", False, Particle),
        KernelArgument("mail_121", False, Particle),
        KernelArgument("mail_122", False, Particle),
        KernelArgument("mail_200", False, Particle),
        KernelArgument("mail_201", False, Particle),
        KernelArgument("mail_202", False, Particle),
        KernelArgument("mail_210", False, Particle),
        KernelArgument("mail_211", False, Particle),
        KernelArgument("mail_212", False, Particle),
        KernelArgument("mail_220", False, Particle),
        KernelArgument("mail_221", False, Particle),
        KernelArgument("mail_222", False, Particle)])

    merge_particles_k = import_kernel("merge_particles_block",
                                      [KernelArgument("from_self", True, Particle),
        KernelArgument("mail000", True, Particle),
        KernelArgument("exists000", True, Type.Int),
        KernelArgument("mail001", True, Particle),
        KernelArgument("exists001", True, Type.Int),
        KernelArgument("mail002", True, Particle),
        KernelArgument("exists002", True, Type.Int),
        KernelArgument("mail010", True, Particle),
        KernelArgument("exists010", True, Type.Int),
        KernelArgument("mail011", True, Particle),
        KernelArgument("exists011", True, Type.Int),
        KernelArgument("mail012", True, Particle),
        KernelArgument("exists012", True, Type.Int),
        KernelArgument("mail020", True, Particle),
        KernelArgument("exists020", True, Type.Int),
        KernelArgument("mail021", True, Particle),
        KernelArgument("exists021", True, Type.Int),
        KernelArgument("mail022", True, Particle),
        KernelArgument("exists022", True, Type.Int),
        KernelArgument("mail100", True, Particle),
        KernelArgument("exists100", True, Type.Int),
        KernelArgument("mail101", True, Particle),
        KernelArgument("exists101", True, Type.Int),
        KernelArgument("mail102", True, Particle),
        KernelArgument("exists102", True, Type.Int),
        KernelArgument("mail110", True, Particle),
        KernelArgument("exists110", True, Type.Int),
        KernelArgument("mail112", True, Particle),
        KernelArgument("exists112", True, Type.Int),
        KernelArgument("mail120", True, Particle),
        KernelArgument("exists120", True, Type.Int),
        KernelArgument("mail121", True, Particle),
        KernelArgument("exists121", True, Type.Int),
        KernelArgument("mail122", True, Particle),
        KernelArgument("exists122", True, Type.Int),
        KernelArgument("mail200", True, Particle),
        KernelArgument("exists200", True, Type.Int),
        KernelArgument("mail201", True, Particle),
        KernelArgument("exists201", True, Type.Int),
        KernelArgument("mail202", True, Particle),
        KernelArgument("exists202", True, Type.Int),
        KernelArgument("mail210", True, Particle),
        KernelArgument("exists210", True, Type.Int),
        KernelArgument("mail211", True, Particle),
        KernelArgument("exists211", True, Type.Int),
        KernelArgument("mail212", True, Particle),
        KernelArgument("exists212", True, Type.Int),
        KernelArgument("mail220", True, Particle),
        KernelArgument("exists220", True, Type.Int),
        KernelArgument("mail221", True, Particle),
        KernelArgument("exists221", True, Type.Int),
        KernelArgument("mail222", True, Particle),
        KernelArgument("exists222", True, Type.Int),
                                       KernelArgument("cell_new", False, Particle)])

    # Race-free per-fragment diagnostic (see particle_diag_block in
    # ucodes.h for why count_particles_k/sum_kinetic_energy_k's old
    # shared-accumulator pattern is gone): each fragment writes into its
    # own distinct cell of n_particles_per_frag/ke_per_frag below, and the
    # whole-domain total is a real sgc sum() reduction over that Array.
    particle_diag_k = import_kernel("particle_diag_block",
                                     [KernelArgument("cell", True, Particle),
                                      KernelArgument("count_out", False, Type.Double, 3),
                                      KernelArgument("ke_out", False, Type.Double, 3)])

    print_step_k = import_kernel("print_step",
                                 [KernelArgument("ts", True, Type.Int),
                                  KernelArgument("n_particles", True, Type.Double),
                                  KernelArgument("ke", True, Type.Double)])

    dump_density_k = import_kernel("dump_density_block",
                                   [KernelArgument("ts", True, Type.Int),
                                    KernelArgument("nts_dump", True, Type.Int),
                                    KernelArgument("ii", True, Type.Int),
                                    KernelArgument("jj", True, Type.Int),
                                    KernelArgument("kk", True, Type.Int),
                                    KernelArgument("bn", True, Type.Int),
                                    KernelArgument("bm", True, Type.Int),
                                    KernelArgument("bl", True, Type.Int),
                                    KernelArgument("ro", True, Type.Double, 3)])

    # -- mesh / fragment grid / physics parameters: matches pic3d_common/
    # common/pic3d.par (the file pic3d_seq.cpp itself reads) exactly for
    # every domain/physics value that has no runnability cost (xn=yn=zn=1.0,
    # cx=cy=cz=0.5, pm=sm=0.5, radius=0.2, w=1.0 plain Jacobi, tau=0.0001,
    # ntp=2, nts=5, np=1,000,000). nfx=nfy=nfz=10 (vs the reference's own
    # 3x3x3): 1000 (10x10x10) fragments, each owning a 40^3 interior, for
    # a 400^3 total grid. Both nfx and nx must divide evenly (pysol's Array
    # requires every fragment to share one block size, unlike the
    # reference's own non-uniform split) and nx must stay even - an odd nx
    # puts a node exactly on the sun at cx=0.5, an exact r=0 that blows up
    # to NaN within one step (confirmed directly: the reference itself does
    # this too at nx=99, cleanly recovers at nx=100 or 102 - the reference's
    # own choice of an even nx was never a free choice).
    #
    # The per-fragment interior (40^3, not the 100^3 the 300^3/3x3x3
    # configuration this was verified against used - see the
    # non-determinism note below) is capped by MEMORY, not by generated
    # code size: 1000 fragments compiles to barely more C++ than 27 did (sgc
    # emits one real nested for-loop per kernel, not one unrolled call per
    # fragment - that part scales fine), but a 1000^3/100^3-interior attempt
    # was OOM-killed (confirmed via dmesg) partway through the INITIAL Jacobi
    # solve: fp alone is ~8.1GB per version at that size x 1000 fragments,
    # and fp's own version counter (it) grows across every single sweep with
    # no explicit release in this code (nor any such hook found in pysol) -
    # 31GB of RAM did not even cover a handful of live fp versions. 40^3
    # keeps the 7 field arrays' combined footprint in the single-digit GB
    # range with real headroom. np[xy]=1130 still lands 1,000,996 particles
    # (matching the reference's own np almost exactly), unchanged - the disk
    # still averages roughly 50 particles per in-disk cell at this
    # resolution. Only nt is still reduced from 100 to 30, since each extra
    # step costs a full fixed-count Jacobi solve over all 1000 fragments;
    # eps's data-driven early stop isn't replicated, see max_jacobi_iters
    # below.
    nfx : Type.Int = 10
    nfy : Type.Int = 10
    nfz : Type.Int = 10
    nfx_last : Type.Int = nfx - 1
    nfy_last : Type.Int = nfy - 1
    nfz_last : Type.Int = nfz - 1

    bn : Type.Int = 42
    bm : Type.Int = 42
    bl : Type.Int = 42

    xn : Type.Double = 1.0
    yn : Type.Double = 1.0
    zn : Type.Double = 1.0
    # xn / 400.0, not "0.0025": safe since xn is a variable, not a literal -
    # see am below for why a literal/literal division must be avoided here.
    hx : Type.Double = xn / 400.0
    hy : Type.Double = yn / 400.0
    hz : Type.Double = zn / 400.0

    cx : Type.Double = 0.5
    cy : Type.Double = 0.5
    cz : Type.Double = 0.5

    pm : Type.Double = 0.5
    sm : Type.Double = 0.5
    radius : Type.Double = 0.2
    npx_particles : Type.Int = 1130
    npy_particles : Type.Int = 1130
    # am = pm / <actual generated count>, confirmed exactly by instrumenting
    # generate_particles_disk_k directly (1000996, matching a hand-computed
    # geometric count too, and deterministic - unlike a diagnostic count
    # taken after any push/merge round, see the module docstring's note on
    # cross-fragment GrowingArray non-determinism), written as "pm / N.0"
    # (N a literal, pm a variable) rather than two literals: the .sol
    # generator renders a literal/literal division's operands without their
    # decimal point when both happen to be whole numbers (1.0 -> "1",
    # 29.0 -> "29"), which silently becomes C++ integer division - safe
    # here only because pm is a variable reference, not a second literal.
    am : Type.Double = pm / 1000996.0

    tau : Type.Double = 0.0001
    w_relax : Type.Double = 1.0

    num_ts : Type.Int = 100
    nts_dump : Type.Int = 5
    max_jacobi_iters : Type.Int = 80

    # -- arrays: every one fragmented nfx x nfy x nfz, 1-cell ghost per side
    # (bn x bm x bl blocks) except the particle GrowingArrays, which have no
    # block dimension at all (see ucodes.h's module comment) --
    ro_raw = Array(3, 3, [nfx, nfy, nfz], [bn, bm, bl], Type.Double)
    ro = Array(3, 3, [nfx, nfy, nfz], [bn, bm, bl], Type.Double)
    fp = Array(3, 3, [nfx, nfy, nfz], [bn, bm, bl], Type.Double)
    fi = Array(3, 3, [nfx, nfy, nfz], [bn, bm, bl], Type.Double)
    fx = Array(3, 3, [nfx, nfy, nfz], [bn, bm, bl], Type.Double)
    fy = Array(3, 3, [nfx, nfy, nfz], [bn, bm, bl], Type.Double)
    fz = Array(3, 3, [nfx, nfy, nfz], [bn, bm, bl], Type.Double)

    # Stage-1 scratch copy for the potential exchange's two-stage ghost
    # exchange (see copy_block3d_k above) - every block's own potential
    # data is copied here first, so exchange_potential_copy_k's neighbor
    # reads (stage 2) never read an array a concurrently-running task
    # might still be writing. Reused (overwritten) for every exchange
    # call, always at version 0 - each stage-1 pass fully completes before
    # its matching stage-2 pass starts reading it, so no further
    # versioning is needed here. The density exchange needed an analogous
    # ro_raw_stable scratch array too, but pysol_update_array_3d_26_full_sum
    # now keeps its own internal scratch, so that array is gone.
    fp_stable = Array(3, 3, [nfx, nfy, nfz], [bn, bm, bl], Type.Double)

    particles = GrowingArray(3, [nfx, nfy, nfz], Particle)
    # Per-fragment diagnostic scratch (see particle_diag_k above) - one
    # cell per fragment, no ghost margin needed.
    n_particles_per_frag = Array(3, 3, [nfx, nfy, nfz], [1, 1, 1], Type.Double)
    ke_per_frag = Array(3, 3, [nfx, nfy, nfz], [1, 1, 1], Type.Double)
    mail_self = GrowingArray(3, [nfx, nfy, nfz], Particle)
    mail_000 = GrowingArray(3, [nfx, nfy, nfz], Particle)
    mail_001 = GrowingArray(3, [nfx, nfy, nfz], Particle)
    mail_002 = GrowingArray(3, [nfx, nfy, nfz], Particle)
    mail_010 = GrowingArray(3, [nfx, nfy, nfz], Particle)
    mail_011 = GrowingArray(3, [nfx, nfy, nfz], Particle)
    mail_012 = GrowingArray(3, [nfx, nfy, nfz], Particle)
    mail_020 = GrowingArray(3, [nfx, nfy, nfz], Particle)
    mail_021 = GrowingArray(3, [nfx, nfy, nfz], Particle)
    mail_022 = GrowingArray(3, [nfx, nfy, nfz], Particle)
    mail_100 = GrowingArray(3, [nfx, nfy, nfz], Particle)
    mail_101 = GrowingArray(3, [nfx, nfy, nfz], Particle)
    mail_102 = GrowingArray(3, [nfx, nfy, nfz], Particle)
    mail_110 = GrowingArray(3, [nfx, nfy, nfz], Particle)
    mail_112 = GrowingArray(3, [nfx, nfy, nfz], Particle)
    mail_120 = GrowingArray(3, [nfx, nfy, nfz], Particle)
    mail_121 = GrowingArray(3, [nfx, nfy, nfz], Particle)
    mail_122 = GrowingArray(3, [nfx, nfy, nfz], Particle)
    mail_200 = GrowingArray(3, [nfx, nfy, nfz], Particle)
    mail_201 = GrowingArray(3, [nfx, nfy, nfz], Particle)
    mail_202 = GrowingArray(3, [nfx, nfy, nfz], Particle)
    mail_210 = GrowingArray(3, [nfx, nfy, nfz], Particle)
    mail_211 = GrowingArray(3, [nfx, nfy, nfz], Particle)
    mail_212 = GrowingArray(3, [nfx, nfy, nfz], Particle)
    mail_220 = GrowingArray(3, [nfx, nfy, nfz], Particle)
    mail_221 = GrowingArray(3, [nfx, nfy, nfz], Particle)
    mail_222 = GrowingArray(3, [nfx, nfy, nfz], Particle)

    # -- initial particles: one lattice-in-disk, each fragment filtering
    # the SAME global candidate lattice down to its own owned box (see
    # ucodes.h's generate_particles_disk_k) - no initial migration needed,
    # and (unlike loading everything onto one fragment) correct regardless
    # of how many fragment-hops separate a particle from fragment (0,0,0) --
    for i in range(nfx):
            for j in range(nfy):
                for k in range(nfz):
                    generate_particles_disk_k(cx, cy, cz, radius, hx, hy, hz,
                                               i, j, k, bn - 2, bm - 2, bl - 2,
                                               npx_particles, npy_particles,
                                               particles[0, i][j][k])

    # -- initial boundary (Dirichlet, point-mass pm at the domain center) --
    for i in range(nfx):
            for j in range(nfy):
                for k in range(nfz):
                    has_lo_x_s1 : Type.Int = 1
                    has_hi_x_s1 : Type.Int = 1
                    has_lo_y_s1 : Type.Int = 1
                    has_hi_y_s1 : Type.Int = 1
                    has_lo_z_s1 : Type.Int = 1
                    has_hi_z_s1 : Type.Int = 1
                    if i == 0:
                        has_lo_x_s1 = 0
                    if i == nfx_last:
                        has_hi_x_s1 = 0
                    if j == 0:
                        has_lo_y_s1 = 0
                    if j == nfy_last:
                        has_hi_y_s1 = 0
                    if k == 0:
                        has_lo_z_s1 = 0
                    if k == nfz_last:
                        has_hi_z_s1 = 0
                    im1_s1 : Type.Int = i - 1
                    ip1_s1 : Type.Int = i + 1
                    jm1_s1 : Type.Int = j - 1
                    jp1_s1 : Type.Int = j + 1
                    km1_s1 : Type.Int = k - 1
                    kp1_s1 : Type.Int = k + 1
                    if i == 0:
                        im1_s1 = i
                    if i == nfx_last:
                        ip1_s1 = i
                    if j == 0:
                        jm1_s1 = j
                    if j == nfy_last:
                        jp1_s1 = j
                    if k == 0:
                        km1_s1 = k
                    if k == nfz_last:
                        kp1_s1 = k
                    exists_000_s1 : Type.Int = has_lo_x_s1 * has_lo_y_s1 * has_lo_z_s1
                    exists_001_s1 : Type.Int = has_lo_x_s1 * has_lo_y_s1
                    exists_002_s1 : Type.Int = has_lo_x_s1 * has_lo_y_s1 * has_hi_z_s1
                    exists_010_s1 : Type.Int = has_lo_x_s1 * has_lo_z_s1
                    exists_011_s1 : Type.Int = has_lo_x_s1
                    exists_012_s1 : Type.Int = has_lo_x_s1 * has_hi_z_s1
                    exists_020_s1 : Type.Int = has_lo_x_s1 * has_hi_y_s1 * has_lo_z_s1
                    exists_021_s1 : Type.Int = has_lo_x_s1 * has_hi_y_s1
                    exists_022_s1 : Type.Int = has_lo_x_s1 * has_hi_y_s1 * has_hi_z_s1
                    exists_100_s1 : Type.Int = has_lo_y_s1 * has_lo_z_s1
                    exists_101_s1 : Type.Int = has_lo_y_s1
                    exists_102_s1 : Type.Int = has_lo_y_s1 * has_hi_z_s1
                    exists_110_s1 : Type.Int = has_lo_z_s1
                    exists_112_s1 : Type.Int = has_hi_z_s1
                    exists_120_s1 : Type.Int = has_hi_y_s1 * has_lo_z_s1
                    exists_121_s1 : Type.Int = has_hi_y_s1
                    exists_122_s1 : Type.Int = has_hi_y_s1 * has_hi_z_s1
                    exists_200_s1 : Type.Int = has_hi_x_s1 * has_lo_y_s1 * has_lo_z_s1
                    exists_201_s1 : Type.Int = has_hi_x_s1 * has_lo_y_s1
                    exists_202_s1 : Type.Int = has_hi_x_s1 * has_lo_y_s1 * has_hi_z_s1
                    exists_210_s1 : Type.Int = has_hi_x_s1 * has_lo_z_s1
                    exists_211_s1 : Type.Int = has_hi_x_s1
                    exists_212_s1 : Type.Int = has_hi_x_s1 * has_hi_z_s1
                    exists_220_s1 : Type.Int = has_hi_x_s1 * has_hi_y_s1 * has_lo_z_s1
                    exists_221_s1 : Type.Int = has_hi_x_s1 * has_hi_y_s1
                    exists_222_s1 : Type.Int = has_hi_x_s1 * has_hi_y_s1 * has_hi_z_s1
                    init_boundary_potential_k(has_lo_x_s1, has_hi_x_s1, has_lo_y_s1, has_hi_y_s1, has_lo_z_s1, has_hi_z_s1,
                                               hx, hy, hz, cx, cy, cz, pm,
                                               i, j, k, bn - 2, bm - 2, bl - 2,
                                               fp[0, i][j][k])

    # -- initial density deposit + 26-neighbor additive exchange --
    for i in range(nfx):
            for j in range(nfy):
                for k in range(nfz):
                    compute_density_raw_k(hx, hy, hz, am, i, j, k, bn - 2, bm - 2, bl - 2,
                                          particles[0, i][j][k], ro_raw[i][j][k])

    # The additive 26-neighbor exchange itself (read raw into a stable
    # scratch copy, then sum every existing neighbor's overlapping slab on
    # top of a self-copy) is now the generic runtime primitive
    # pysol_update_array_3d_26_full_sum - see
    # src/pysol/exchange_3d_26_full_sum.sol - which replaces the
    # copy_block3d_k + exchange_density_additive_k two-stage call pair this
    # site used before.
    pysol_update_array_3d_26_full_sum(ro_raw, 0, ro, 0)

    # -- initial Poisson solve (plain weighted Jacobi, w=1.0 - the reference's
    # own default - so no SOR/Gauss-Seidel approximation is needed here; "it"
    # keeps growing across every sweep AND every later timestep's solve too,
    # reusing the latest potential as the next solve's warm-started initial
    # guess, exactly like pysol's other pic3d port's "it" counter) --
    it : Type.Int = 0
    for sweep in range(max_jacobi_iters):
            for i in range(nfx):
                for j in range(nfy):
                    for k in range(nfz):
                        poisson_jacobi_sweep_k(hx, hy, hz, w_relax, ro[i][j][k], fp[it, i][j][k],
                                               fp[it + 1, i][j][k])
            # Stage 1 ("read"): snapshot every block's own fresh fp[it+1]
            # before the exchange below reads any of them as a neighbor.
            for i in range(nfx):
                for j in range(nfy):
                    for k in range(nfz):
                        copy_block3d_k(bn, bm, bl, fp[it + 1, i][j][k], fp_stable[it, i][j][k])
            # Stage 2 ("write"): every neighbor read below comes from
            # fp_stable (fully settled by stage 1), never fp[it+1] itself.
            for i in range(nfx):
                for j in range(nfy):
                    for k in range(nfz):
                        has_lo_x_s3 : Type.Int = 1
                        has_hi_x_s3 : Type.Int = 1
                        has_lo_y_s3 : Type.Int = 1
                        has_hi_y_s3 : Type.Int = 1
                        has_lo_z_s3 : Type.Int = 1
                        has_hi_z_s3 : Type.Int = 1
                        if i == 0:
                            has_lo_x_s3 = 0
                        if i == nfx_last:
                            has_hi_x_s3 = 0
                        if j == 0:
                            has_lo_y_s3 = 0
                        if j == nfy_last:
                            has_hi_y_s3 = 0
                        if k == 0:
                            has_lo_z_s3 = 0
                        if k == nfz_last:
                            has_hi_z_s3 = 0
                        im1_s3 : Type.Int = i - 1
                        ip1_s3 : Type.Int = i + 1
                        jm1_s3 : Type.Int = j - 1
                        jp1_s3 : Type.Int = j + 1
                        km1_s3 : Type.Int = k - 1
                        kp1_s3 : Type.Int = k + 1
                        if i == 0:
                            im1_s3 = i
                        if i == nfx_last:
                            ip1_s3 = i
                        if j == 0:
                            jm1_s3 = j
                        if j == nfy_last:
                            jp1_s3 = j
                        if k == 0:
                            km1_s3 = k
                        if k == nfz_last:
                            kp1_s3 = k
                        exists_000_s3 : Type.Int = has_lo_x_s3 * has_lo_y_s3 * has_lo_z_s3
                        exists_001_s3 : Type.Int = has_lo_x_s3 * has_lo_y_s3
                        exists_002_s3 : Type.Int = has_lo_x_s3 * has_lo_y_s3 * has_hi_z_s3
                        exists_010_s3 : Type.Int = has_lo_x_s3 * has_lo_z_s3
                        exists_011_s3 : Type.Int = has_lo_x_s3
                        exists_012_s3 : Type.Int = has_lo_x_s3 * has_hi_z_s3
                        exists_020_s3 : Type.Int = has_lo_x_s3 * has_hi_y_s3 * has_lo_z_s3
                        exists_021_s3 : Type.Int = has_lo_x_s3 * has_hi_y_s3
                        exists_022_s3 : Type.Int = has_lo_x_s3 * has_hi_y_s3 * has_hi_z_s3
                        exists_100_s3 : Type.Int = has_lo_y_s3 * has_lo_z_s3
                        exists_101_s3 : Type.Int = has_lo_y_s3
                        exists_102_s3 : Type.Int = has_lo_y_s3 * has_hi_z_s3
                        exists_110_s3 : Type.Int = has_lo_z_s3
                        exists_112_s3 : Type.Int = has_hi_z_s3
                        exists_120_s3 : Type.Int = has_hi_y_s3 * has_lo_z_s3
                        exists_121_s3 : Type.Int = has_hi_y_s3
                        exists_122_s3 : Type.Int = has_hi_y_s3 * has_hi_z_s3
                        exists_200_s3 : Type.Int = has_hi_x_s3 * has_lo_y_s3 * has_lo_z_s3
                        exists_201_s3 : Type.Int = has_hi_x_s3 * has_lo_y_s3
                        exists_202_s3 : Type.Int = has_hi_x_s3 * has_lo_y_s3 * has_hi_z_s3
                        exists_210_s3 : Type.Int = has_hi_x_s3 * has_lo_z_s3
                        exists_211_s3 : Type.Int = has_hi_x_s3
                        exists_212_s3 : Type.Int = has_hi_x_s3 * has_hi_z_s3
                        exists_220_s3 : Type.Int = has_hi_x_s3 * has_hi_y_s3 * has_lo_z_s3
                        exists_221_s3 : Type.Int = has_hi_x_s3 * has_hi_y_s3
                        exists_222_s3 : Type.Int = has_hi_x_s3 * has_hi_y_s3 * has_hi_z_s3
                        exchange_potential_copy_k(fp[it + 1, i][j][k],
                                                     fp_stable[it, im1_s3][jm1_s3][km1_s3], exists_000_s3,
                                                     fp_stable[it, im1_s3][jm1_s3][k], exists_001_s3,
                                                     fp_stable[it, im1_s3][jm1_s3][kp1_s3], exists_002_s3,
                                                     fp_stable[it, im1_s3][j][km1_s3], exists_010_s3,
                                                     fp_stable[it, im1_s3][j][k], exists_011_s3,
                                                     fp_stable[it, im1_s3][j][kp1_s3], exists_012_s3,
                                                     fp_stable[it, im1_s3][jp1_s3][km1_s3], exists_020_s3,
                                                     fp_stable[it, im1_s3][jp1_s3][k], exists_021_s3,
                                                     fp_stable[it, im1_s3][jp1_s3][kp1_s3], exists_022_s3,
                                                     fp_stable[it, i][jm1_s3][km1_s3], exists_100_s3,
                                                     fp_stable[it, i][jm1_s3][k], exists_101_s3,
                                                     fp_stable[it, i][jm1_s3][kp1_s3], exists_102_s3,
                                                     fp_stable[it, i][j][km1_s3], exists_110_s3,
                                                     fp_stable[it, i][j][kp1_s3], exists_112_s3,
                                                     fp_stable[it, i][jp1_s3][km1_s3], exists_120_s3,
                                                     fp_stable[it, i][jp1_s3][k], exists_121_s3,
                                                     fp_stable[it, i][jp1_s3][kp1_s3], exists_122_s3,
                                                     fp_stable[it, ip1_s3][jm1_s3][km1_s3], exists_200_s3,
                                                     fp_stable[it, ip1_s3][jm1_s3][k], exists_201_s3,
                                                     fp_stable[it, ip1_s3][jm1_s3][kp1_s3], exists_202_s3,
                                                     fp_stable[it, ip1_s3][j][km1_s3], exists_210_s3,
                                                     fp_stable[it, ip1_s3][j][k], exists_211_s3,
                                                     fp_stable[it, ip1_s3][j][kp1_s3], exists_212_s3,
                                                     fp_stable[it, ip1_s3][jp1_s3][km1_s3], exists_220_s3,
                                                     fp_stable[it, ip1_s3][jp1_s3][k], exists_221_s3,
                                                     fp_stable[it, ip1_s3][jp1_s3][kp1_s3], exists_222_s3,
                                                   fp[it + 2, i][j][k])
            it = it + 2

    # -- add the sun's potential, then the force (acceleration, G=1) --
    for i in range(nfx):
            for j in range(nfy):
                for k in range(nfz):
                    add_sun_potential_k(hx, hy, hz, cx, cy, cz, sm, i, j, k, bn - 2, bm - 2, bl - 2,
                                        fp[it, i][j][k], fi[i][j][k])
                    compute_force_k(hx, hy, hz, fi[i][j][k], fx[i][j][k], fy[i][j][k], fz[i][j][k])

    # -- main loop: push+27-way migrate, deposit+exchange density, solve,
    # add sun potential + force, diagnostics, periodic density dump --
    pv : Type.Int = 0
    for ts in range(num_ts):
            for i in range(nfx):
                for j in range(nfy):
                    for k in range(nfz):
                        has_lo_x_s4 : Type.Int = 1
                        has_hi_x_s4 : Type.Int = 1
                        has_lo_y_s4 : Type.Int = 1
                        has_hi_y_s4 : Type.Int = 1
                        has_lo_z_s4 : Type.Int = 1
                        has_hi_z_s4 : Type.Int = 1
                        if i == 0:
                            has_lo_x_s4 = 0
                        if i == nfx_last:
                            has_hi_x_s4 = 0
                        if j == 0:
                            has_lo_y_s4 = 0
                        if j == nfy_last:
                            has_hi_y_s4 = 0
                        if k == 0:
                            has_lo_z_s4 = 0
                        if k == nfz_last:
                            has_hi_z_s4 = 0
                        im1_s4 : Type.Int = i - 1
                        ip1_s4 : Type.Int = i + 1
                        jm1_s4 : Type.Int = j - 1
                        jp1_s4 : Type.Int = j + 1
                        km1_s4 : Type.Int = k - 1
                        kp1_s4 : Type.Int = k + 1
                        if i == 0:
                            im1_s4 = i
                        if i == nfx_last:
                            ip1_s4 = i
                        if j == 0:
                            jm1_s4 = j
                        if j == nfy_last:
                            jp1_s4 = j
                        if k == 0:
                            km1_s4 = k
                        if k == nfz_last:
                            kp1_s4 = k
                        exists_000_s4 : Type.Int = has_lo_x_s4 * has_lo_y_s4 * has_lo_z_s4
                        exists_001_s4 : Type.Int = has_lo_x_s4 * has_lo_y_s4
                        exists_002_s4 : Type.Int = has_lo_x_s4 * has_lo_y_s4 * has_hi_z_s4
                        exists_010_s4 : Type.Int = has_lo_x_s4 * has_lo_z_s4
                        exists_011_s4 : Type.Int = has_lo_x_s4
                        exists_012_s4 : Type.Int = has_lo_x_s4 * has_hi_z_s4
                        exists_020_s4 : Type.Int = has_lo_x_s4 * has_hi_y_s4 * has_lo_z_s4
                        exists_021_s4 : Type.Int = has_lo_x_s4 * has_hi_y_s4
                        exists_022_s4 : Type.Int = has_lo_x_s4 * has_hi_y_s4 * has_hi_z_s4
                        exists_100_s4 : Type.Int = has_lo_y_s4 * has_lo_z_s4
                        exists_101_s4 : Type.Int = has_lo_y_s4
                        exists_102_s4 : Type.Int = has_lo_y_s4 * has_hi_z_s4
                        exists_110_s4 : Type.Int = has_lo_z_s4
                        exists_112_s4 : Type.Int = has_hi_z_s4
                        exists_120_s4 : Type.Int = has_hi_y_s4 * has_lo_z_s4
                        exists_121_s4 : Type.Int = has_hi_y_s4
                        exists_122_s4 : Type.Int = has_hi_y_s4 * has_hi_z_s4
                        exists_200_s4 : Type.Int = has_hi_x_s4 * has_lo_y_s4 * has_lo_z_s4
                        exists_201_s4 : Type.Int = has_hi_x_s4 * has_lo_y_s4
                        exists_202_s4 : Type.Int = has_hi_x_s4 * has_lo_y_s4 * has_hi_z_s4
                        exists_210_s4 : Type.Int = has_hi_x_s4 * has_lo_z_s4
                        exists_211_s4 : Type.Int = has_hi_x_s4
                        exists_212_s4 : Type.Int = has_hi_x_s4 * has_hi_z_s4
                        exists_220_s4 : Type.Int = has_hi_x_s4 * has_hi_y_s4 * has_lo_z_s4
                        exists_221_s4 : Type.Int = has_hi_x_s4 * has_hi_y_s4
                        exists_222_s4 : Type.Int = has_hi_x_s4 * has_hi_y_s4 * has_hi_z_s4
                        push_particles_k(tau, xn, yn, zn, hx, hy, hz, i, j, k, bn - 2, bm - 2, bl - 2,
                                         fx[pv, i][j][k], fy[pv, i][j][k], fz[pv, i][j][k],
                                         particles[pv, i][j][k],
                                         mail_self[pv + 1, i][j][k],
                                         mail_000[pv + 1, i][j][k],
                                         mail_001[pv + 1, i][j][k],
                                         mail_002[pv + 1, i][j][k],
                                         mail_010[pv + 1, i][j][k],
                                         mail_011[pv + 1, i][j][k],
                                         mail_012[pv + 1, i][j][k],
                                         mail_020[pv + 1, i][j][k],
                                         mail_021[pv + 1, i][j][k],
                                         mail_022[pv + 1, i][j][k],
                                         mail_100[pv + 1, i][j][k],
                                         mail_101[pv + 1, i][j][k],
                                         mail_102[pv + 1, i][j][k],
                                         mail_110[pv + 1, i][j][k],
                                         mail_112[pv + 1, i][j][k],
                                         mail_120[pv + 1, i][j][k],
                                         mail_121[pv + 1, i][j][k],
                                         mail_122[pv + 1, i][j][k],
                                         mail_200[pv + 1, i][j][k],
                                         mail_201[pv + 1, i][j][k],
                                         mail_202[pv + 1, i][j][k],
                                         mail_210[pv + 1, i][j][k],
                                         mail_211[pv + 1, i][j][k],
                                         mail_212[pv + 1, i][j][k],
                                         mail_220[pv + 1, i][j][k],
                                         mail_221[pv + 1, i][j][k],
                                         mail_222[pv + 1, i][j][k])

            for i in range(nfx):
                for j in range(nfy):
                    for k in range(nfz):
                        has_lo_x_s5 : Type.Int = 1
                        has_hi_x_s5 : Type.Int = 1
                        has_lo_y_s5 : Type.Int = 1
                        has_hi_y_s5 : Type.Int = 1
                        has_lo_z_s5 : Type.Int = 1
                        has_hi_z_s5 : Type.Int = 1
                        if i == 0:
                            has_lo_x_s5 = 0
                        if i == nfx_last:
                            has_hi_x_s5 = 0
                        if j == 0:
                            has_lo_y_s5 = 0
                        if j == nfy_last:
                            has_hi_y_s5 = 0
                        if k == 0:
                            has_lo_z_s5 = 0
                        if k == nfz_last:
                            has_hi_z_s5 = 0
                        im1_s5 : Type.Int = i - 1
                        ip1_s5 : Type.Int = i + 1
                        jm1_s5 : Type.Int = j - 1
                        jp1_s5 : Type.Int = j + 1
                        km1_s5 : Type.Int = k - 1
                        kp1_s5 : Type.Int = k + 1
                        if i == 0:
                            im1_s5 = i
                        if i == nfx_last:
                            ip1_s5 = i
                        if j == 0:
                            jm1_s5 = j
                        if j == nfy_last:
                            jp1_s5 = j
                        if k == 0:
                            km1_s5 = k
                        if k == nfz_last:
                            kp1_s5 = k
                        exists_000_s5 : Type.Int = has_lo_x_s5 * has_lo_y_s5 * has_lo_z_s5
                        exists_001_s5 : Type.Int = has_lo_x_s5 * has_lo_y_s5
                        exists_002_s5 : Type.Int = has_lo_x_s5 * has_lo_y_s5 * has_hi_z_s5
                        exists_010_s5 : Type.Int = has_lo_x_s5 * has_lo_z_s5
                        exists_011_s5 : Type.Int = has_lo_x_s5
                        exists_012_s5 : Type.Int = has_lo_x_s5 * has_hi_z_s5
                        exists_020_s5 : Type.Int = has_lo_x_s5 * has_hi_y_s5 * has_lo_z_s5
                        exists_021_s5 : Type.Int = has_lo_x_s5 * has_hi_y_s5
                        exists_022_s5 : Type.Int = has_lo_x_s5 * has_hi_y_s5 * has_hi_z_s5
                        exists_100_s5 : Type.Int = has_lo_y_s5 * has_lo_z_s5
                        exists_101_s5 : Type.Int = has_lo_y_s5
                        exists_102_s5 : Type.Int = has_lo_y_s5 * has_hi_z_s5
                        exists_110_s5 : Type.Int = has_lo_z_s5
                        exists_112_s5 : Type.Int = has_hi_z_s5
                        exists_120_s5 : Type.Int = has_hi_y_s5 * has_lo_z_s5
                        exists_121_s5 : Type.Int = has_hi_y_s5
                        exists_122_s5 : Type.Int = has_hi_y_s5 * has_hi_z_s5
                        exists_200_s5 : Type.Int = has_hi_x_s5 * has_lo_y_s5 * has_lo_z_s5
                        exists_201_s5 : Type.Int = has_hi_x_s5 * has_lo_y_s5
                        exists_202_s5 : Type.Int = has_hi_x_s5 * has_lo_y_s5 * has_hi_z_s5
                        exists_210_s5 : Type.Int = has_hi_x_s5 * has_lo_z_s5
                        exists_211_s5 : Type.Int = has_hi_x_s5
                        exists_212_s5 : Type.Int = has_hi_x_s5 * has_hi_z_s5
                        exists_220_s5 : Type.Int = has_hi_x_s5 * has_hi_y_s5 * has_lo_z_s5
                        exists_221_s5 : Type.Int = has_hi_x_s5 * has_hi_y_s5
                        exists_222_s5 : Type.Int = has_hi_x_s5 * has_hi_y_s5 * has_hi_z_s5
                        merge_particles_k(mail_self[pv + 1, i][j][k],
                                           mail_222[pv + 1, im1_s5][jm1_s5][km1_s5], exists_000_s5,
                                           mail_221[pv + 1, im1_s5][jm1_s5][k], exists_001_s5,
                                           mail_220[pv + 1, im1_s5][jm1_s5][kp1_s5], exists_002_s5,
                                           mail_212[pv + 1, im1_s5][j][km1_s5], exists_010_s5,
                                           mail_211[pv + 1, im1_s5][j][k], exists_011_s5,
                                           mail_210[pv + 1, im1_s5][j][kp1_s5], exists_012_s5,
                                           mail_202[pv + 1, im1_s5][jp1_s5][km1_s5], exists_020_s5,
                                           mail_201[pv + 1, im1_s5][jp1_s5][k], exists_021_s5,
                                           mail_200[pv + 1, im1_s5][jp1_s5][kp1_s5], exists_022_s5,
                                           mail_122[pv + 1, i][jm1_s5][km1_s5], exists_100_s5,
                                           mail_121[pv + 1, i][jm1_s5][k], exists_101_s5,
                                           mail_120[pv + 1, i][jm1_s5][kp1_s5], exists_102_s5,
                                           mail_112[pv + 1, i][j][km1_s5], exists_110_s5,
                                           mail_110[pv + 1, i][j][kp1_s5], exists_112_s5,
                                           mail_102[pv + 1, i][jp1_s5][km1_s5], exists_120_s5,
                                           mail_101[pv + 1, i][jp1_s5][k], exists_121_s5,
                                           mail_100[pv + 1, i][jp1_s5][kp1_s5], exists_122_s5,
                                           mail_022[pv + 1, ip1_s5][jm1_s5][km1_s5], exists_200_s5,
                                           mail_021[pv + 1, ip1_s5][jm1_s5][k], exists_201_s5,
                                           mail_020[pv + 1, ip1_s5][jm1_s5][kp1_s5], exists_202_s5,
                                           mail_012[pv + 1, ip1_s5][j][km1_s5], exists_210_s5,
                                           mail_011[pv + 1, ip1_s5][j][k], exists_211_s5,
                                           mail_010[pv + 1, ip1_s5][j][kp1_s5], exists_212_s5,
                                           mail_002[pv + 1, ip1_s5][jp1_s5][km1_s5], exists_220_s5,
                                           mail_001[pv + 1, ip1_s5][jp1_s5][k], exists_221_s5,
                                           mail_000[pv + 1, ip1_s5][jp1_s5][kp1_s5], exists_222_s5,
                                           particles[pv + 2, i][j][k])

            for i in range(nfx):
                for j in range(nfy):
                    for k in range(nfz):
                        compute_density_raw_k(hx, hy, hz, am, i, j, k, bn - 2, bm - 2, bl - 2,
                                              particles[pv + 2, i][j][k], ro_raw[pv, i][j][k])

            # 26-neighbor additive exchange - see the comment at the
            # initial-density call site above.
            pysol_update_array_3d_26_full_sum(ro_raw, pv, ro, pv + 1)

            for sweep in range(max_jacobi_iters):
                for i in range(nfx):
                    for j in range(nfy):
                        for k in range(nfz):
                            poisson_jacobi_sweep_k(hx, hy, hz, w_relax, ro[pv + 1, i][j][k], fp[it, i][j][k],
                                                   fp[it + 1, i][j][k])
                # Stage 1 ("read"): snapshot every block's own fresh
                # fp[it+1] before the exchange below reads any of them as
                # a neighbor.
                for i in range(nfx):
                    for j in range(nfy):
                        for k in range(nfz):
                            copy_block3d_k(bn, bm, bl, fp[it + 1, i][j][k], fp_stable[it, i][j][k])
                # Stage 2 ("write"): every neighbor read below comes from
                # fp_stable, never fp[it+1] itself.
                for i in range(nfx):
                    for j in range(nfy):
                        for k in range(nfz):
                            has_lo_x_s7 : Type.Int = 1
                            has_hi_x_s7 : Type.Int = 1
                            has_lo_y_s7 : Type.Int = 1
                            has_hi_y_s7 : Type.Int = 1
                            has_lo_z_s7 : Type.Int = 1
                            has_hi_z_s7 : Type.Int = 1
                            if i == 0:
                                has_lo_x_s7 = 0
                            if i == nfx_last:
                                has_hi_x_s7 = 0
                            if j == 0:
                                has_lo_y_s7 = 0
                            if j == nfy_last:
                                has_hi_y_s7 = 0
                            if k == 0:
                                has_lo_z_s7 = 0
                            if k == nfz_last:
                                has_hi_z_s7 = 0
                            im1_s7 : Type.Int = i - 1
                            ip1_s7 : Type.Int = i + 1
                            jm1_s7 : Type.Int = j - 1
                            jp1_s7 : Type.Int = j + 1
                            km1_s7 : Type.Int = k - 1
                            kp1_s7 : Type.Int = k + 1
                            if i == 0:
                                im1_s7 = i
                            if i == nfx_last:
                                ip1_s7 = i
                            if j == 0:
                                jm1_s7 = j
                            if j == nfy_last:
                                jp1_s7 = j
                            if k == 0:
                                km1_s7 = k
                            if k == nfz_last:
                                kp1_s7 = k
                            exists_000_s7 : Type.Int = has_lo_x_s7 * has_lo_y_s7 * has_lo_z_s7
                            exists_001_s7 : Type.Int = has_lo_x_s7 * has_lo_y_s7
                            exists_002_s7 : Type.Int = has_lo_x_s7 * has_lo_y_s7 * has_hi_z_s7
                            exists_010_s7 : Type.Int = has_lo_x_s7 * has_lo_z_s7
                            exists_011_s7 : Type.Int = has_lo_x_s7
                            exists_012_s7 : Type.Int = has_lo_x_s7 * has_hi_z_s7
                            exists_020_s7 : Type.Int = has_lo_x_s7 * has_hi_y_s7 * has_lo_z_s7
                            exists_021_s7 : Type.Int = has_lo_x_s7 * has_hi_y_s7
                            exists_022_s7 : Type.Int = has_lo_x_s7 * has_hi_y_s7 * has_hi_z_s7
                            exists_100_s7 : Type.Int = has_lo_y_s7 * has_lo_z_s7
                            exists_101_s7 : Type.Int = has_lo_y_s7
                            exists_102_s7 : Type.Int = has_lo_y_s7 * has_hi_z_s7
                            exists_110_s7 : Type.Int = has_lo_z_s7
                            exists_112_s7 : Type.Int = has_hi_z_s7
                            exists_120_s7 : Type.Int = has_hi_y_s7 * has_lo_z_s7
                            exists_121_s7 : Type.Int = has_hi_y_s7
                            exists_122_s7 : Type.Int = has_hi_y_s7 * has_hi_z_s7
                            exists_200_s7 : Type.Int = has_hi_x_s7 * has_lo_y_s7 * has_lo_z_s7
                            exists_201_s7 : Type.Int = has_hi_x_s7 * has_lo_y_s7
                            exists_202_s7 : Type.Int = has_hi_x_s7 * has_lo_y_s7 * has_hi_z_s7
                            exists_210_s7 : Type.Int = has_hi_x_s7 * has_lo_z_s7
                            exists_211_s7 : Type.Int = has_hi_x_s7
                            exists_212_s7 : Type.Int = has_hi_x_s7 * has_hi_z_s7
                            exists_220_s7 : Type.Int = has_hi_x_s7 * has_hi_y_s7 * has_lo_z_s7
                            exists_221_s7 : Type.Int = has_hi_x_s7 * has_hi_y_s7
                            exists_222_s7 : Type.Int = has_hi_x_s7 * has_hi_y_s7 * has_hi_z_s7
                            exchange_potential_copy_k(fp[it + 1, i][j][k],
                                                         fp_stable[it, im1_s7][jm1_s7][km1_s7], exists_000_s7,
                                                         fp_stable[it, im1_s7][jm1_s7][k], exists_001_s7,
                                                         fp_stable[it, im1_s7][jm1_s7][kp1_s7], exists_002_s7,
                                                         fp_stable[it, im1_s7][j][km1_s7], exists_010_s7,
                                                         fp_stable[it, im1_s7][j][k], exists_011_s7,
                                                         fp_stable[it, im1_s7][j][kp1_s7], exists_012_s7,
                                                         fp_stable[it, im1_s7][jp1_s7][km1_s7], exists_020_s7,
                                                         fp_stable[it, im1_s7][jp1_s7][k], exists_021_s7,
                                                         fp_stable[it, im1_s7][jp1_s7][kp1_s7], exists_022_s7,
                                                         fp_stable[it, i][jm1_s7][km1_s7], exists_100_s7,
                                                         fp_stable[it, i][jm1_s7][k], exists_101_s7,
                                                         fp_stable[it, i][jm1_s7][kp1_s7], exists_102_s7,
                                                         fp_stable[it, i][j][km1_s7], exists_110_s7,
                                                         fp_stable[it, i][j][kp1_s7], exists_112_s7,
                                                         fp_stable[it, i][jp1_s7][km1_s7], exists_120_s7,
                                                         fp_stable[it, i][jp1_s7][k], exists_121_s7,
                                                         fp_stable[it, i][jp1_s7][kp1_s7], exists_122_s7,
                                                         fp_stable[it, ip1_s7][jm1_s7][km1_s7], exists_200_s7,
                                                         fp_stable[it, ip1_s7][jm1_s7][k], exists_201_s7,
                                                         fp_stable[it, ip1_s7][jm1_s7][kp1_s7], exists_202_s7,
                                                         fp_stable[it, ip1_s7][j][km1_s7], exists_210_s7,
                                                         fp_stable[it, ip1_s7][j][k], exists_211_s7,
                                                         fp_stable[it, ip1_s7][j][kp1_s7], exists_212_s7,
                                                         fp_stable[it, ip1_s7][jp1_s7][km1_s7], exists_220_s7,
                                                         fp_stable[it, ip1_s7][jp1_s7][k], exists_221_s7,
                                                         fp_stable[it, ip1_s7][jp1_s7][kp1_s7], exists_222_s7,
                                                       fp[it + 2, i][j][k])
                it = it + 2

            for i in range(nfx):
                for j in range(nfy):
                    for k in range(nfz):
                        add_sun_potential_k(hx, hy, hz, cx, cy, cz, sm, i, j, k, bn - 2, bm - 2, bl - 2,
                                            fp[it, i][j][k], fi[i][j][k])
                        compute_force_k(hx, hy, hz, fi[i][j][k], fx[pv + 1, i][j][k], fy[pv + 1, i][j][k], fz[pv + 1, i][j][k])

            for i in range(nfx):
                for j in range(nfy):
                    for k in range(nfz):
                        particle_diag_k(particles[pv + 2, i][j][k],
                                         n_particles_per_frag[0, i][j][k],
                                         ke_per_frag[0, i][j][k])
            n_particles : Type.Double = sum(n_particles_per_frag)
            ke : Type.Double = sum(ke_per_frag)
            print_step_k(ts, n_particles, ke)

            for i in range(nfx):
                for j in range(nfy):
                    for k in range(nfz):
                        dump_density_k(ts, nts_dump, i, j, k, bn - 2, bm - 2, bl - 2, ro[pv + 1, i][j][k])

            pv = pv + 2



def _read_doubles(path, count):
    a = array.array("d")
    with open(path, "rb") as fh:
        a.frombytes(fh.read(count * 8))
    return a


def write_density_files():
    """Post-processing step, run after the simulation: gathers every
    fragment's dump_density_block file for each dumped timestep (one every
    NTS_DUMP steps, matching pic3d.par's own "nts" - see dump_density_block
    in ucodes.h for the matching ts%nts_dump gate) and assembles
    results/ro_<ts>.raw (a plain, flat NX*NY*NZ float64 binary of the
    merged global density, row-major [i][j][k]) - no in-program gather, so
    this works unchanged under MPI (each rank only ever writes the files
    for the fragments it owns), exactly like pysol's other pic3d port's
    write_vti_files()/update_array_3d_loop_test.py's verify().

    Also builds the three sum-projection GIFs pic3d_common's own
    save_array_sxy/sxz/syz would write as a stack of per-step files -
    here as one animated .gif per projection instead (results/ro_sxy.gif,
    ro_sxz.gif, ro_syz.gif), since that's far easier to actually look at
    than a directory of individual frames.

    In addition, for every dumped timestep this also writes the exact
    file set pic3d_seq.cpp's own save_array_sxy/sxz/syz/cxy/cxz/cyz would
    (see pic3d_common/common/mesh3d.h's mesh2d_log2d/mesh3d_log2d_xy/xz/yz):
    results/ro.sxy.NNNNNN, ro.sxz.NNNNNN, ro.syz.NNNNNN (same sum-
    projections as the GIF frames, just also dumped to disk one file per
    step) and results/ro.cxy.NNNNNN, ro.cxz.NNNNNN, ro.cyz.NNNNNN (a single
    slice through the plane that passes through the domain center
    (cx,cy,cz), picked the same way the reference does via coord2cell -
    here just an index lookup since the full merged array is already in
    hand). Each is a flat, row-major float64 binary (nx*ny, nx*nz or
    ny*nz doubles - gnuplot's own `binary array=(nx,ny) format="%lf"`
    reads them directly, matching pic3d_common/sequential/gif.plt),
    exactly like the reference's own files - this does NOT replace the
    .raw/.bin/.gif pipeline above, which stays as pic3d_v2's own primary
    (and easier to actually view) output.
    """
    import numpy as np

    nx, ny, nz = NFX * INTERIOR, NFY * INTERIOR, NFZ * INTERIOR
    lx0 = int(round(CX / HX))
    ly0 = int(round(CY / HY))
    lz0 = int(round(CZ / HZ))
    frame_ts = []
    frames_sxy, frames_sxz, frames_syz = [], [], []

    for ts in range(0, NUM_TS, NTS_DUMP):
        glob = array.array("d", [0.0]) * (nx * ny * nz)
        found_any = False
        for ii in range(NFX):
            for jj in range(NFY):
                for kk in range(NFZ):
                    path = f"results/ro_{ts:05d}_{ii}_{jj}_{kk}.bin"
                    if not os.path.exists(path):
                        continue
                    found_any = True
                    rec = _read_doubles(path, INTERIOR * INTERIOR * INTERIOR)
                    idx = 0
                    for li in range(INTERIOR):
                        for lj in range(INTERIOR):
                            for lk in range(INTERIOR):
                                gi, gj, gk = ii * INTERIOR + li, jj * INTERIOR + lj, kk * INTERIOR + lk
                                glob[(gi * ny + gj) * nz + gk] = rec[idx]
                                idx += 1
        if not found_any:
            continue
        with open(f"results/ro_{ts:05d}.raw", "wb") as out:
            out.write(glob.tobytes())

        arr = np.frombuffer(glob, dtype=np.float64).reshape(nx, ny, nz)
        frame_ts.append(ts)
        sxy = arr.sum(axis=2)  # sum over Z -> XY projection
        sxz = arr.sum(axis=1)  # sum over Y -> XZ projection
        syz = arr.sum(axis=0)  # sum over X -> YZ projection
        frames_sxy.append(sxy)
        frames_sxz.append(sxz)
        frames_syz.append(syz)

        # pic3d_seq.cpp-matching file set (see the docstring above) -
        # row-major float64, same (x*ny+y)/(x*nz+z)/(y*nz+z) flattening
        # mesh2d_log2d/mesh3d_log2d_* use, which is numpy's default
        # C-order flatten of these (nx,ny)/(nx,nz)/(ny,nz) arrays.
        with open(f"results/ro.sxy.{ts:06d}", "wb") as out:
            out.write(sxy.tobytes())
        with open(f"results/ro.sxz.{ts:06d}", "wb") as out:
            out.write(sxz.tobytes())
        with open(f"results/ro.syz.{ts:06d}", "wb") as out:
            out.write(syz.tobytes())
        with open(f"results/ro.cxy.{ts:06d}", "wb") as out:
            out.write(np.ascontiguousarray(arr[:, :, lz0]).tobytes())
        with open(f"results/ro.cxz.{ts:06d}", "wb") as out:
            out.write(np.ascontiguousarray(arr[:, ly0, :]).tobytes())
        with open(f"results/ro.cyz.{ts:06d}", "wb") as out:
            out.write(np.ascontiguousarray(arr[lx0, :, :]).tobytes())

    if not frame_ts:
        return

    make_density_gif(frames_sxy, frame_ts, "results/ro_sxy.gif", "X", "Y", "Z")
    make_density_gif(frames_sxz, frame_ts, "results/ro_sxz.gif", "X", "Z", "Y")
    make_density_gif(frames_syz, frame_ts, "results/ro_syz.gif", "Y", "Z", "X")


def make_density_gif(frames, frame_ts, out_path, xlabel, ylabel, summed_axis_label):
    """Renders a list of 2D sum-projection frames (one per dumped
    timestep) into a single animated GIF, color-scaled consistently
    across every frame so brightness changes reflect real density
    evolution rather than per-frame rescaling."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    vmin = min(f.min() for f in frames)
    vmax = max(f.max() for f in frames)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    im = ax.imshow(frames[0].T, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title = ax.set_title(f"ro, summed over {summed_axis_label}  (ts={frame_ts[0]})")
    fig.colorbar(im, ax=ax, label="sum(ro)")
    fig.tight_layout()

    def update(i):
        im.set_data(frames[i].T)
        title.set_text(f"ro, summed over {summed_axis_label}  (ts={frame_ts[i]})")
        return im, title

    anim = FuncAnimation(fig, update, frames=len(frames), interval=500, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=2))
    plt.close(fig)


if __name__ == "__main__":
    # Run with OMP_NUM_THREADS=1 at this file's full nfx=nfy=nfz=10/
    # bn=bm=bl=42/np=1,000,996 scale - see the module docstring's updated
    # non-determinism note (bug (2), NOT fixed): confirmed real particle-
    # data corruption under multithreading at this scale specifically,
    # concentrated in the z=cz fragment layer every particle starts in.
    # Smaller configurations (this repo's own tests) were not observed to
    # be affected by that one; bug (1) (a diagnostic-only accumulator
    # race) IS fixed and no longer needs single-threading to work around.
    # See pic3d_v2/pic3d_common's own README/TODO for the reference's
    # own, separately-acknowledged "too fast particles" limitation
    # (particles near the central singularity gaining enough velocity to
    # skip more than one fragment per step) - this port's
    # push_particles_block has its own CFL clamp specifically to avoid
    # that, unrelated to either bug above.
    load_config("config.json")

    print("==========PIC3D_V2 (pysol, self-gravitating N-body)============")
    pysol_compile("pic3d_pysol")
    pysol_run("pic3d_pysol")
    write_density_files()

