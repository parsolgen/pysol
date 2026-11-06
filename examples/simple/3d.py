
import os
import sys
from pysol import Array, KernelArgument, Type, import_kernel, pysol, pysol_compile, pysol_run, load_config, pysol_update_array, sol_generator


@pysol
def solve3d():
    # ── kernel imports ────────────────────────────────────────────────────────
    hello_world_k = import_kernel("hello_world_k", [])

    # ── body ──────────────────────────────────────────────────────────────────
    hello_world_k()
    nb : Type.Int = 10
    mb : Type.Int = 10
    lb : Type.Int = 10

    block_n_ : Type.Int = 10
    block_m_ : Type.Int = 10
    block_l_ : Type.Int = 10

    A = Array(3, 3, [nb, mb, lb], [block_n_, block_m_, block_l_], Type.Double)

    init_el_k = import_kernel("init_el3d",
                              [KernelArgument("ii", True, Type.Int),
                               KernelArgument("jj", True, Type.Int),
                               KernelArgument("kk", True, Type.Int),
                               KernelArgument("A", False, Type.Double, 3)])
    print_el_k = import_kernel("print_el3d",
                              [KernelArgument("ii", True, Type.Int),
                               KernelArgument("jj", True, Type.Int),
                               KernelArgument("kk", True, Type.Int),
                               KernelArgument("A", False, Type.Double, 3)])
    for i in range(nb):
        for j in range(mb):
            for k in range(lb):
                init_el_k(i, j, k, A[i][j][k])
    hello_world_k()
    pysol_update_array(A, 0, A, 1)
    for i in range(nb):
        for j in range(mb):
            for k in range(lb):
                print_el_k(i, j, k, A[1, i][j][k])

if __name__ == "__main__":
    load_config("config.json")
    # Call the function — the decorator prints the analysis summary first.
    #hello_world()

    # Print the generated .sol file.
    #sol_path = os.path.join(
    #    os.path.dirname(os.path.abspath(__file__)), "hello_world.sol"
    #)
    #print("\n" + "=" * 60)
    #print(f"Generated: {sol_path}")
    #print("=" * 60)
    #if not os.path.exists("hello_world.so"):
    #pysol_compile("hello_world")
    #pysol_run("hello_world")

    print("==========SOLVER============")
    pysol_compile("solve3d")
    pysol_run("solve3d")