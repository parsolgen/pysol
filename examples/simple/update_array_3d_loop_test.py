
import os
import sys
from pysol import Array, KernelArgument, Type, import_kernel, pysol, pysol_compile, pysol_run, load_config, pysol_update_array, sol_generator

# Same block layout as update_array_3d_test.py, but here pysol_update_array
# is called once per iteration of a 50-iteration while loop instead of
# once: each iteration first modifies the array (modify_ghost_test3d adds 1
# to every interior cell) and then exchanges ghosts into the next version,
# incrementing the version number every time.
# NB/MB/LB/NUM_ITERS/STEP must stay in sync with the literals inside
# update_array_3d_loop_test() below — the @pysol AST analysis only
# understands literals local to the decorated function.
NB = 4
MB = 4
LB = 4
NUM_ITERS = 50
STEP = 1.0


@pysol
def update_array_3d_loop_test():
    # ── kernel imports ────────────────────────────────────────────────────────
    hello_world_k = import_kernel("hello_world_k", [])

    init_k = import_kernel("init_ghost_test3d",
                            [KernelArgument("ii", True, Type.Int),
                             KernelArgument("jj", True, Type.Int),
                             KernelArgument("kk", True, Type.Int),
                             KernelArgument("A", False, Type.Double, 3)])
    modify_k = import_kernel("modify_ghost_test3d",
                             [KernelArgument("ii", True, Type.Int),
                             KernelArgument("jj", True, Type.Int),
                             KernelArgument("kk", True, Type.Int),
                             KernelArgument("A", True, Type.Double, 3),
                             KernelArgument("A_out", False, Type.Double, 3)])
    check_k = import_kernel("check_ghost_test3d_offset",
                             [KernelArgument("ii", True, Type.Int),
                              KernelArgument("jj", True, Type.Int),
                              KernelArgument("kk", True, Type.Int),
                              KernelArgument("nb", True, Type.Int),
                              KernelArgument("mb", True, Type.Int),
                              KernelArgument("lb", True, Type.Int),
                              KernelArgument("offset", True, Type.Double),
                              KernelArgument("A", True, Type.Double, 3)])
    print_el_k = import_kernel("print_el3d",
                              [KernelArgument("ii", True, Type.Int),
                               KernelArgument("jj", True, Type.Int),
                               KernelArgument("kk", True, Type.Int),
                               KernelArgument("A", True, Type.Double, 3)])

    # ── body ──────────────────────────────────────────────────────────────────
    hello_world_k()

    nb : Type.Int = 4
    mb : Type.Int = 4
    lb : Type.Int = 4

    block_n_ : Type.Int = 3
    block_m_ : Type.Int = 3
    block_l_ : Type.Int = 3

    A = Array(3, 3, [nb, mb, lb], [block_n_ + 2, block_m_ + 2, block_l_ + 2], Type.Double)

    for i in range(nb):
        for j in range(mb):
            for k in range(lb):
                init_k(i, j, k, A[i][j][k])

    # Each of the 50 iterations: modify every block's interior at the
    # current version, then exchange ghosts into version it+1.
    it : Type.Int = 0
    while it < 100:
        for i in range(nb):
            for j in range(mb):
                for k in range(lb):
                    modify_k(i, j, k, A[it, i][j][k], A[it + 1, i][j][k])
        pysol_update_array(A, it + 1, A, it + 2)
        it = it + 2

    # After 50 increments of 1.0 each, every interior cell (and, via the
    # halo exchange, every matching ghost cell) should be offset by 50.0
    # relative to the plain block-id scheme check_ghost_test3d_offset uses.
    # NB: read version "it" here, not a fixed/older literal - A's backing
    # storage is a 5-slot ring buffer (nversions=5, confirmed in the
    # generated .cpp), so any version label more than a couple of steps
    # behind "it" resolves to a slot that has since been overwritten by a
    # later iteration rather than the fully exchanged final state.
    offset : Type.Double = 50.0
    for i in range(nb):
        for j in range(mb):
            for k in range(lb):
                print_el_k(i, j, k, A[it, i][j][k])
                check_k(i, j, k, nb, mb, lb, offset, A[it, i][j][k])


def verify():
    """Read the per-block results written by check_ghost_test3d_offset and
    assert that 100 iterations of modify-then-exchange produced no
    mismatches."""
    failures = []
    for i in range(NB):
        for j in range(MB):
            for k in range(LB):
                path = f"check3d_loop_{i}_{j}_{k}"
                with open(path) as fh:
                    mismatches, unset = (int(v) for v in fh.read().split())
                if mismatches or unset:
                    failures.append((i, j, k, mismatches, unset))

    if failures:
        print(f"FAIL: {len(failures)}/{NB * MB * LB} block(s) with issues: {failures}")
        sys.exit(1)

    print(f"PASS: pysol_update_array survived {NUM_ITERS} modify+exchange "
          f"iterations across {NB * MB * LB} blocks")


if __name__ == "__main__":
    load_config("config.json")

    print("==========UPDATE ARRAY 3D LOOP TEST============")
    pysol_compile("update_array_3d_loop_test")
    pysol_run("update_array_3d_loop_test")
    verify()
