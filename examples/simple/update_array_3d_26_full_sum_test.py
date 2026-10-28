
import os
import sys
from pysol import Array, KernelArgument, Type, import_kernel, pysol, pysol_compile, pysol_run, load_config, pysol_update_array_3d_26_full_sum, sol_generator

# Block layout: 3x3x3 blocks, each block a full 5x5x5 (no separate ghost
# layer - pysol_update_array_3d_26_full_sum is for the OVERLAPPING-node
# scheme used by examples/pic3d_v2, where a block's own 2-cell-wide border
# slab physically overlaps its neighbors'). 3x3x3 blocks is the smallest
# layout that exercises every one of the 26 directions on at least one
# block (the center block (1,1,1) has all 26 neighbors; the corner/edge/
# face blocks each have some directions missing at the domain boundary).
# NB/MB/LB must stay in sync with the nb/mb/lb literals inside
# update_array_3d_26_full_sum_test() below - the @pysol AST analysis only
# understands literals local to the decorated function, not module-level
# names.
NB = 3
MB = 3
LB = 3


@pysol
def update_array_3d_26_full_sum_test():
    # ── kernel imports ────────────────────────────────────────────────────────
    hello_world_k = import_kernel("hello_world_k", [])

    init_k = import_kernel("init_ghost_test3d_sum26",
                            [KernelArgument("ii", True, Type.Int),
                             KernelArgument("jj", True, Type.Int),
                             KernelArgument("kk", True, Type.Int),
                             KernelArgument("A", False, Type.Double, 3)])
    check_k = import_kernel("check_ghost_test3d_sum26",
                             [KernelArgument("ii", True, Type.Int),
                              KernelArgument("jj", True, Type.Int),
                              KernelArgument("kk", True, Type.Int),
                              KernelArgument("nb", True, Type.Int),
                              KernelArgument("mb", True, Type.Int),
                              KernelArgument("lb", True, Type.Int),
                              KernelArgument("A", True, Type.Double, 3)])
    # ── body ──────────────────────────────────────────────────────────────────
    hello_world_k()

    nb : Type.Int = 3
    mb : Type.Int = 3
    lb : Type.Int = 3

    block_n_ : Type.Int = 5
    block_m_ : Type.Int = 5
    block_l_ : Type.Int = 5

    A = Array(3, 3, [nb, mb, lb], [block_n_, block_m_, block_l_], Type.Double)

    for i in range(nb):
        for j in range(mb):
            for k in range(lb):
                init_k(i, j, k, A[i][j][k])

    pysol_update_array_3d_26_full_sum(A, 0, A, 1)

    for i in range(nb):
        for j in range(mb):
            for k in range(lb):
                check_k(i, j, k, nb, mb, lb, A[1, i][j][k])


def verify():
    """Read the per-block results written by check_ghost_test3d_sum26 and
    assert that pysol_update_array_3d_26_full_sum's additive 26-neighbor
    exchange produced no mismatches."""
    failures = []
    for i in range(NB):
        for j in range(MB):
            for k in range(LB):
                path = f"check3d_sum26_{i}_{j}_{k}"
                with open(path) as fh:
                    mismatches = int(fh.read().strip())
                if mismatches:
                    failures.append((i, j, k, mismatches))

    if failures:
        print(f"FAIL: {len(failures)}/{NB * MB * LB} block(s) with issues: {failures}")
        sys.exit(1)

    print(f"PASS: pysol_update_array_3d_26_full_sum verified across {NB * MB * LB} blocks")


if __name__ == "__main__":
    load_config("config.json")

    print("==========UPDATE ARRAY 3D 26 FULL SUM TEST============")
    pysol_compile("update_array_3d_26_full_sum_test")
    pysol_run("update_array_3d_26_full_sum_test")
    verify()
