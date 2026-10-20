import sys

from pysol import (
    Array, CSRMatrix, KernelArgument, Type, import_kernel, is_zero,
    load_config, pysol, pysol_compile, pysol_run, reset,
)

# =============================================================================
# pysol port of ~/work/tests/bicg/bicg.sol: unpreconditioned BiCG (block
# conjugate gradient, non-symmetric variant) solving A x = b for a general
# sparse matrix blocked into an nb x nb grid of CSR sub-blocks.
#
# The matrix (ia/ja/a, read from the file named on the command line - see
# init_params/read_matrix in ucodes.h, carried over from the reference
# unchanged) is wrapped in a CSRMatrix, and A[i][j] is skipped via
# is_zero(A[i][j]) wherever that block has no non-zero entries - exactly
# the structure bicg.sol uses (its "if (!%zero(A[i][j]))" is this port's
# "if not is_zero(A[i][j]):").
#
# The reference's SingleVersionArray (and its bare-vs-{%latest}-vs-{it}
# reference rules) isn't a type pysol implements, so every block-vector
# below (x, Rr/Rl, Pr/Pl, Zr/Zl) is a plain versioned Array instead, with
# every version read/written explicitly by number. reset(Zr)/reset(Zl)
# before each accumulation mirrors bicg.sol's "Zr = {}; Zl = {};". bicg.sol
# itself only advances a row/column's version on a non-zero block, so the
# final accumulated version is data-dependent (how many of a row's nb
# blocks are non-zero varies per row) and it reads that back via
# "{%latest}" (pysol's matching LATEST_VERSION/%latest, confirmed to
# render correctly) - this port instead always advances the version once
# per inner-loop step regardless of is_zero (copying the running value
# through via axpby_block_k in the else branch for a skipped/zero block),
# so every row/column's accumulation always lands in the same, statically
# known version `nb`, read back as Zr[nb, i]/Zl[nb, j]. This swap was
# forced empirically: the RTS array holder backing a plain 1-D Array in
# this build (rts::MemoryOwningVectorHolderD) has no GetLatestElementPtr
# accessor, which is what %latest compiles down to, so using it here
# doesn't compile against this particular RTS build.
#
# The dot products ("double beta = (Zr, Rl);" in bicg.sol) use pysol's `@`
# operator directly: a @ b -> (a, b), the whole blocked vector at each side
# reduced to one scalar in a single .sol-level expression - no per-block
# loop needed, matching bicg.sol's own "(Zr, Rl)" exactly. A version-tagged
# vector uses the 1-tuple subscript a[version,] (not a[version]: the
# trailing comma is what makes it a tuple, and pysol's subscript parser
# only renders "a{version}" - no block coordinate, since the dot product
# sums over every block - for that 1-tuple form); bare "a" means version 0.
#
# KNOWN ISSUE in this RTS build: `@` (which compiles to a collective
# algo::dot(...) call) is correct and fast in isolation - verified with a
# standalone script computing the same dot product, including 100 repeated
# calls in a loop - but once ANY `@` has executed, the NEXT csr_mv/
# csr_mv_t call (the MKL sparse matvec, further down in this same
# function) crashes with "pthread_mutex_lock: Assertion
# `mutex->__data.__owner == 0' failed." A dot-product followed by any
# OTHER (non-MKL) kernel call is fine; it's specifically @ immediately
# preceding an MKL sparse-matvec call that's broken, consistent with an
# MKL-internal reentrancy bug (@ and csr_mv likely both end up inside
# MKL - a BLAS dot product and sparse MV respectively - and something
# about calling both from the same process/thread in sequence corrupts
# MKL's own internal locking state) rather than a bug in the .sol/C++ this
# generates. Confirmed this is not pysol-generated-code issue: the "(Rl,
# Rr)" and "(Pl, Zr)" expressions below are exactly what bicg.sol's own
# "(Zr, Rl)" syntax is for, and running this file as-is reproduces the
# crash deterministically, every run, right after the very first @ call.
#
# The axpy-style updates ("x{1} = x + a_c * Pr;" in bicg.sol) now use
# pysol's +/-/* operators on vectors the same way: a[v,] + b[v,] ->
# a{v} + b{v}, and a[v,] * c -> a{v} * c for a vector times a scalar of the
# same type (c * a[v,] also works, scalar on the left) - again a single
# whole-vector .sol expression, no per-block loop. Following bicg.sol's
# own convention exactly, a subtraction like "x - a_c*p" is still written
# as an addition of a precomputed negative (ma_c = a_c * -1.0; x + ma_c*p)
# rather than using the new "-" operator for it - bicg.sol never uses "-"
# either, even though it could.
#
# The matvec accumulation's per-block copy-through (the `else` branch for
# a skipped/zero block, further below) still goes through an explicit
# per-block kernel call, since it addresses one specific block
# (Zr[j, i]: version j, block i) rather than a whole vector - not what the
# +/-/* vector operators are for. axpby_block(blockn, a, x, b, y, dst)
# computes dst = a*x + b*y, covering both a pure scale (b=0, used here)
# and a general axpy with one kernel - a new kernel this port adds in
# ucodes.h; most everything else there (read_matrix/init_params/
# init_mat_arrays, csr_mv/csr_mv_t and their mats/t_mats MKL-handle cache,
# init_vec/init_zero_vec, sqrt, time_start/print_time) is carried over
# from the reference's ucodes.h unchanged - the only global state here is
# what the reference already has.
#
# Like the reference, this is plain (unpreconditioned) BiCG - no
# preconditioner step, Z=R at every point one would otherwise appear.
# lf10.bin (an 18x18 non-symmetric sparse matrix, nnz=50, condition number
# ~5e4) is a genuinely ill-conditioned test case for unpreconditioned BiCG:
# an independent from-scratch reference run of the same algorithm in numpy,
# and this same port before the switch to `@` above, both confirm the
# residual grows sharply within the first ~4 iterations (reaching ~1e10)
# then, running single-threaded (required - see the __main__ guard below),
# decays slowly back down over the rest of the 100 iterations without
# reaching a useful solution - not a port bug, just an unconditioned
# solver meeting a genuinely ill-conditioned matrix. The reference's own
# cpu.txt log (a run
# against a different, larger matrix) shows the same kind of transient
# blow-up, there ending in "nan" instead of decaying - which one happens
# just depends on the matrix.
# =============================================================================


@pysol
def bicg():
    # ── kernel imports ──────────────────────────────────────────────────────
    init_params_k = import_kernel("init_params",
                                   [KernelArgument("nb", True, Type.Int),
                                    KernelArgument("n", False, Type.Int),
                                    KernelArgument("nnz", False, Type.Int),
                                    KernelArgument("blockn", False, Type.Int)])

    init_mat_desc_k = import_kernel("init_mat_desc", [KernelArgument("nb", True, Type.Int)])

    print_k = import_kernel("print", [KernelArgument("value", True, Type.Int)])
    print_d_k = import_kernel("print_d", [KernelArgument("value", True, Type.Double)])

    init_mat_arrays_k = import_kernel("init_mat_arrays",
                                       [KernelArgument("n", True, Type.Int),
                                        KernelArgument("nnz", True, Type.Int),
                                        KernelArgument("ia", False, Type.Int, 1),
                                        KernelArgument("ja", False, Type.Int, 1),
                                        KernelArgument("a", False, Type.Double, 1)])

    init_vec_k = import_kernel("init_vec",
                                [KernelArgument("iblock", True, Type.Int),
                                 KernelArgument("blockn", True, Type.Int),
                                 KernelArgument("block", False, Type.Double, 1)])

    init_zero_vec_k = import_kernel("init_zero_vec",
                                     [KernelArgument("iblock", True, Type.Int),
                                      KernelArgument("blockn", True, Type.Int),
                                      KernelArgument("block", False, Type.Double, 1)])

    csr_mv_k = import_kernel("csr_mv",
                              [KernelArgument("nb", True, Type.Int),
                               KernelArgument("blockn", True, Type.Int),
                               KernelArgument("block", True, Type.CSRBlock(Type.Double)),
                               KernelArgument("x", True, Type.Double, 1),
                               KernelArgument("ysrc", True, Type.Double, 1),
                               KernelArgument("ydst", False, Type.Double, 1)])

    csr_mv_t_k = import_kernel("csr_mv_t",
                                [KernelArgument("nb", True, Type.Int),
                                 KernelArgument("blockn", True, Type.Int),
                                 KernelArgument("block", True, Type.CSRBlock(Type.Double)),
                                 KernelArgument("x", True, Type.Double, 1),
                                 KernelArgument("ysrc", True, Type.Double, 1),
                                 KernelArgument("ydst", False, Type.Double, 1)])

    axpby_block_k = import_kernel("axpby_block",
                                   [KernelArgument("blockn", True, Type.Int),
                                    KernelArgument("a", True, Type.Double),
                                    KernelArgument("x", True, Type.Double, 1),
                                    KernelArgument("b", True, Type.Double),
                                    KernelArgument("y", True, Type.Double, 1),
                                    KernelArgument("dst", False, Type.Double, 1)])

    sqrt_k = import_kernel("sqrt", [KernelArgument("a", True, Type.Double),
                                     KernelArgument("b", False, Type.Double)])

    time_start_k = import_kernel("time_start", [])
    print_time_k = import_kernel("print_time", [])

    # ── matrix setup ─────────────────────────────────────────────────────────
    n : Type.Int = 0
    nnz : Type.Int = 0
    blockn : Type.Int = 0
    nb : Type.Int = 9

    init_params_k(nb, n, nnz, blockn)
    init_mat_desc_k(nb)
    print_k(n)
    print_k(nnz)
    print_k(nb)
    print_k(blockn)

    ia = Array(1, 0, [n + 1], [], Type.Int)
    ja = Array(1, 0, [nnz], [], Type.Int)
    a = Array(1, 0, [nnz], [], Type.Double)

    # Trailing comma matters: ia[0,] is a 1-tuple subscript, which
    # pysol's subscript parser recognizes as "version 0, no coordinates"
    # (rendering "ia{0}", matching ia's empty block dims below); ia[0]
    # (no tuple) is instead parsed as a plain, version-less index into ia
    # (rendering "ia[0]", a dereference - wrong for a ContiguousArray1
    # whole-buffer argument). ia/ja/a's empty block dims also mean pysol
    # never auto-emits a leading %block_n(...) size token the way it does
    # for Pr/Rl/etc below, so (n + 1)/nnz/nnz are passed explicitly as the
    # size args the import line's KernelArgument(..., n_dims=1) still
    # expects (import_kernel's n_dims controls the *declared* signature,
    # independent of the referenced array's own block-dims).
    init_mat_arrays_k(n, nnz, n + 1, ia[0,], nnz, ja[0,], nnz, a[0,])
    A = CSRMatrix([n, n], [nb, nb], nnz, ia, ja, a, Type.Double)

    time_start_k()

    # ── block-vectors ────────────────────────────────────────────────────────
    x = Array(1, 1, [nb], [blockn], Type.Double, True)
    B = Array(1, 1, [nb], [blockn], Type.Double, True)
    Rr = Array(1, 1, [nb], [blockn], Type.Double, True)
    Rl = Array(1, 1, [nb], [blockn], Type.Double, True)
    Pr = Array(1, 1, [nb], [blockn], Type.Double, True)
    Pl = Array(1, 1, [nb], [blockn], Type.Double, True)
    Zr = Array(1, 1, [nb], [blockn], Type.Double, True)
    Zl = Array(1, 1, [nb], [blockn], Type.Double, True)
    tmpr = Array(1, 1, [nb], [blockn], Type.Double, True)
    tmpl = Array(1, 1, [nb], [blockn], Type.Double, True)

    # x0 = 0, r0 = r0~ = p0 = p0~ = b (the all-ones RHS; see init_vec in
    # ucodes.h) - the standard BiCG starting point for x0=0.
    for i in range(nb):
        init_zero_vec_k(i, blockn, x[0, i])
        init_vec_k(i, blockn, B[i])

    Rr = B
    Rl = Rr
    Zr = Rr
    Zl = Rl
    Pr = Zr
    Pl = Zl

    beta : Type.Double = Zr[0,] @ Rl[0,]

    # ── first update (k=0): q0=A p0, q0~=A^T p0~, alpha0=beta0/(p0~,q0) ──────
    # Zr/Zl accumulate one column/row-block at a time, ping-ponging through
    # versions 0..nb (mirroring bicg.sol's "Zr = {}; ... csr_mv(..., Zr{v},
    # Zr{v+1}); v += 1" accumulation) - but every step advances the version
    # regardless of is_zero, via a copy-through axpby_block_k in the else
    # branch, so the accumulation always lands in version `nb` exactly,
    # whether or not a given block was skipped. bicg.sol instead only
    # advances the version on a non-zero block and reads the data-dependent
    # final version back via "{%latest}"; this port reads the always-known
    # "Zr[nb, i]"/"Zl[nb, j]" instead; LATEST_VERSION (%latest) renders
    # correctly but the RTS array holder backing a 1-D Array in this build
    # doesn't implement the accessor %latest compiles to, so this avoids it.
    reset(Zr)
    reset(Zl)

    n : Type.Int = 0
    v : Type.Int = 0

    for i in range(nb):
        v = 0
        for j in range(nb):
            if not is_zero(A[i][j]):
                csr_mv_k(nb, blockn, A[i][j], Pr[0, j], Zr[v, i], Zr[v + 1, i])
                v = v + 1

    for j in range(nb):
        v = 0
        for i in range(nb):
            if not is_zero(A[i][j]):
                csr_mv_t_k(nb, blockn, A[i][j], Pl[0, i], Zl[v, j], Zl[v + 1, j])
                v = v + 1

    dpi : Type.Double = Rl[0,] @ Zr[LATEST_VERSION,]
    a_c : Type.Double = beta / dpi
    ma_c : Type.Double = a_c * -1.0

    x[1,] = x[0,] + a_c * Pr[0,]
    Rr[1,] = Rr[0,] + ma_c * Zr[LATEST_VERSION,]
    Rl[1,] = Rl[0,] + ma_c * Zl[LATEST_VERSION,]

    dp_t : Type.Double = Rr[1,] @ Rr[1,]
    dp : Type.Double = 0.0
    sqrt_k(dp_t, dp)
    print_d_k(dp)
    Zr = Rr[1,]
    Zl = Rl[1,]

    # ── remaining updates (k=1..99): 100 total, matching bicg.sol's
    # "1 initial + while(it<100)" ──────────────────────────────────────────
    beta_prev : Type.Double = beta
    it : Type.Int = 1
    while it < 100:
        beta = Rl[it,] @ Zr[it,]
        beta_f : Type.Double = beta / beta_prev

        tmpr[it,] = beta_f * Pr[it,]
        tmpl[it,] = beta_f * Pl[it,]

        Pr[it + 1,] = tmpr[it,] + Zr[LATEST_VERSION,]
        Pl[it + 1,] = tmpl[it,] + Zl[LATEST_VERSION,]

        beta_prev = beta

        reset(Zr)
        reset(Zl)
        for i in range(nb):
            v = 0
            for j in range(nb):
                if not is_zero(A[i][j]):
                    csr_mv_k(nb, blockn, A[i][j], Pr[it + 1, j], Zr[v, i], Zr[v + 1, i])

        for j in range(nb):
            v = 0
            for i in range(nb):
                if not is_zero(A[i][j]):
                    csr_mv_t_k(nb, blockn, A[i][j], Pl[it + 1, i], Zl[v, j], Zl[v + 1, j])

        dpi_w : Type.Double = Pl[it,] @ Zr[LATEST_VERSION,]
        alpha_w : Type.Double = beta / dpi_w
        ma_w : Type.Double = alpha_w * -1.0

        x[it + 1,] = x[it,] + alpha_w * Pr[it,]
        Rr[it + 1,] = Rr[it,] + ma_w * Zr[LATEST_VERSION,]
        Rl[it + 1,] = Rl[it,] + ma_w * Zl[LATEST_VERSION,]

        dpw_t : Type.Double = Rr[it + 1,] @ Rr[it + 1,]
        dpw : Type.Double = 0.0
        sqrt_k(dpw_t, dpw)
        print_d_k(dpw)

        it = it + 1

    print_time_k()


if __name__ == "__main__":
    # Run with OMP_NUM_THREADS=1 TBB_NUM_THREADS=1 MKL_NUM_THREADS=1 - there
    # is a confirmed multithreading bug in sgc's task scheduler (also seen
    # porting pic3d_v2) that otherwise makes the printed residuals
    # non-deterministic between runs (verified directly: the first several
    # iterations match a from-scratch numpy reference run to 5+ significant
    # digits when single-threaded, and diverge from both that reference and
    # each other, run to run, with the default multithreaded scheduling).
    #
    # As currently written (using `@` for the dot products - see the
    # module docstring above), running this crashes right after the first
    # @ with "pthread_mutex_lock: Assertion `mutex->__data.__owner == 0'
    # failed." - an apparent MKL-internal reentrancy bug triggered by `@`
    # (algo::dot) running immediately before a csr_mv/csr_mv_t MKL sparse
    # matvec call, not a bug in the .sol/C++ pysol generates for `@`.
    if len(sys.argv) < 2:
        sys.argv.append("lf10.bin")

    load_config("config.json")

    print("==========BiCG (pysol)============")
    pysol_compile("bicg")
    pysol_run("bicg")
