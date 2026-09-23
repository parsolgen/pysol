
import os
import sys
from pysol import Array, KernelArgument, Type, import_kernel, pysol, pysol_compile, pysol_run, load_config, pysol_update_array, sol_generator


@pysol
def update_array_3d_test():
    # ── kernel imports ────────────────────────────────────────────────────────
    hello_world_k = import_kernel("hello_world_k", [])
    print_value_k = import_kernel("printv", [KernelArgument("v", True, Type.Double)])

    init_k = import_kernel("init_el_3d",
                            [KernelArgument("ii", True, Type.Int),
                             KernelArgument("jj", True, Type.Int),
                             KernelArgument("kk", True, Type.Int),
                             KernelArgument("A", False, Type.Double, 3)])

    # ── body ──────────────────────────────────────────────────────────────────
    hello_world_k()

    nb : Type.Int = 10
    mb : Type.Int = 10
    lb : Type.Int = 10

    block_n_ : Type.Int = 10
    block_m_ : Type.Int = 10
    block_l_ : Type.Int = 10

    A = Array(3, 3, [nb, mb, lb], [block_n_, block_m_, block_l_], Type.Double)

    for i in range(nb):
        for j in range(mb):
            for k in range(lb):
                init_k(i, j, k, A[i][j][k])

    value : Type.Double = sum(A)
    print_value_k(value)


if __name__ == "__main__":
    load_config("config.json")

    print("==========REDUCE ARRAY 3D TEST============")
    pysol_compile("update_array_3d_test")
    pysol_run("update_array_3d_test")
