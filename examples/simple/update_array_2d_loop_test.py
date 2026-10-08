
import os
import sys
from pysol import Array, KernelArgument, Type, import_kernel, pysol, pysol_compile, pysol_run, load_config, pysol_update_array, sol_generator

# 2D analogue of update_array_3d_loop_test.py: same block layout idea (NBxMB
# blocks, each block a 1-cell ghost layer around a 3x3 interior), but
# pysol_update_array is exercised on a 2D-blocked 2D array instead of a
# 3D-blocked 3D one. Each iteration of a 100-iteration while loop first
# modifies the array (modify_ghost_test2d adds 1 to every interior cell)
# and then exchanges ghosts into the next version, incrementing the
# version number every time.
# NB/MB/NUM_ITERS/STEP must stay in sync with the literals inside
# update_array_2d_loop_test() below — the @pysol AST analysis only
# understands literals local to the decorated function.
NB = 4
MB = 4
NUM_ITERS = 50
STEP = 1.0


@pysol
def update_array_2d_loop_test():
    # ── kernel imports ────────────────────────────────────────────────────────
    hello_world_k = import_kernel("hello_world_k", [])

    init_k = import_kernel("init_ghost_test2d",
                            [KernelArgument("ii", True, Type.Int),
                             KernelArgument("jj", True, Type.Int),
                             KernelArgument("A", False, Type.Double, 2)])
    modify_k = import_kernel("modify_ghost_test2d",
                             [KernelArgument("ii", True, Type.Int),
                             KernelArgument("jj", True, Type.Int),
                             KernelArgument("A", True, Type.Double, 2),
                             KernelArgument("A_out", False, Type.Double, 2)])
    check_k = import_kernel("check_ghost_test2d_offset",
                             [KernelArgument("ii", True, Type.Int),
                              KernelArgument("jj", True, Type.Int),
                              KernelArgument("nb", True, Type.Int),
                              KernelArgument("mb", True, Type.Int),
                              KernelArgument("offset", True, Type.Double),
                              KernelArgument("A", True, Type.Double, 2)])
    print_el_k = import_kernel("print_el2d",
                              [KernelArgument("ii", True, Type.Int),
                               KernelArgument("jj", True, Type.Int),
                               KernelArgument("A", True, Type.Double, 2)])

    # ── body ──────────────────────────────────────────────────────────────────
    hello_world_k()

    nb : Type.Int = 4
    mb : Type.Int = 4

    block_n_ : Type.Int = 3
    block_m_ : Type.Int = 3

    A = Array(2, 2, [nb, mb], [block_n_ + 2, block_m_ + 2], Type.Double)

    for i in range(nb):
        for j in range(mb):
            init_k(i, j, A[i][j])

    # Each of the 50 iterations: modify every block's interior at the
    # current version, then exchange ghosts into version it+1.
    it : Type.Int = 0
    while it < 100:
        for i in range(nb):
            for j in range(mb):
                modify_k(i, j, A[it, i][j], A[it + 1, i][j])
        pysol_update_array(A, it + 1, A, it + 2)
        it = it + 2

    # After 50 increments of 1.0 each, every interior cell (and, via the
    # halo exchange, every matching ghost cell) should be offset by 50.0
    # relative to the plain block-id scheme check_ghost_test2d_offset uses.
    # NB: read version "it" here, not "it - 2" - A's backing storage is a
    # 5-slot ring buffer (nversions=5, confirmed in the generated .cpp), so
    # "it - 2" resolves to a slot 2 versions stale rather than the fully
    # exchanged final state "it" left behind.
    offset : Type.Double = 50.0
    for i in range(nb):
        for j in range(mb):
            print_el_k(i, j, A[it, i][j])
            check_k(i, j, nb, mb, offset, A[it, i][j])


def verify():
    """Read the per-block results written by check_ghost_test2d_offset and
    assert that 50 iterations of modify-then-exchange produced no
    mismatches."""
    failures = []
    for i in range(NB):
        for j in range(MB):
            path = f"check2d_loop_{i}_{j}"
            with open(path) as fh:
                mismatches, unset = (int(v) for v in fh.read().split())
            if mismatches or unset:
                failures.append((i, j, mismatches, unset))

    if failures:
        print(f"FAIL: {len(failures)}/{NB * MB} block(s) with issues: {failures}")
        sys.exit(1)

    print(f"PASS: pysol_update_array survived {NUM_ITERS} modify+exchange "
          f"iterations across {NB * MB} blocks")


if __name__ == "__main__":
    load_config("config.json")

    print("==========UPDATE ARRAY 2D LOOP TEST============")
    pysol_compile("update_array_2d_loop_test")
    pysol_run("update_array_2d_loop_test")
    verify()
