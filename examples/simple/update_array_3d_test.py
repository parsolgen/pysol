
import os
import sys
from pysol import Array, KernelArgument, Type, import_kernel, pysol, pysol_compile, pysol_run, load_config, pysol_update_array, sol_generator

# Block layout: 4x4x4 blocks, each block 6x6x6 (a 1-cell ghost layer on every
# face around a 4x4x4 interior). Kept small on purpose — this is a
# correctness check for pysol_update_array's 3D halo exchange, not a solver.
# NB/MB/LB must stay in sync with the nb/mb/lb literals inside
# update_array_3d_test() below — the @pysol AST analysis only understands
# literals local to the decorated function, not module-level names.
NB = 4
MB = 4
LB = 4


@pysol
def update_array_3d_test():
    # ── kernel imports ────────────────────────────────────────────────────────
    hello_world_k = import_kernel("hello_world_k", [])

    init_k = import_kernel("init_ghost_test3d",
                            [KernelArgument("ii", True, Type.Int),
                             KernelArgument("jj", True, Type.Int),
                             KernelArgument("kk", True, Type.Int),
                             KernelArgument("A", False, Type.Double, 3)])
    check_k = import_kernel("check_ghost_test3d",
                             [KernelArgument("ii", True, Type.Int),
                              KernelArgument("jj", True, Type.Int),
                              KernelArgument("kk", True, Type.Int),
                              KernelArgument("nb", True, Type.Int),
                              KernelArgument("mb", True, Type.Int),
                              KernelArgument("lb", True, Type.Int),
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

    pysol_update_array(A, 0, A, 1)  

    for i in range(nb):
        for j in range(mb):
            for k in range(lb):
                check_k(i, j, k, nb, mb, lb, A[1, i][j][k])
                print_el_k(i, j, k, A[1, i][j][k])


def verify():
    """Read the per-block results written by check_ghost_test3d and assert
    that pysol_update_array's 3D halo exchange produced no mismatches."""
    failures = []
    for i in range(NB):
        for j in range(MB):
            for k in range(LB):
                path = f"check3d_{i}_{j}_{k}"
                with open(path) as fh:
                    mismatches, unset = (int(v) for v in fh.read().split())
                if mismatches or unset:
                    failures.append((i, j, k, mismatches, unset))

    if failures:
        print(f"FAIL: {len(failures)}/{NB * MB * LB} block(s) with issues: {failures}")
        sys.exit(1)

    print(f"PASS: pysol_update_array 3D halo exchange verified across {NB * MB * LB} blocks")


if __name__ == "__main__":
    load_config("config.json")

    print("==========UPDATE ARRAY 3D TEST============")
    pysol_compile("update_array_3d_test")
    pysol_run("update_array_3d_test")
    verify()
