
import os
import sys
from pysol import Array, KernelArgument, Type, import_kernel, pysol, pysol_compile, pysol_run, load_config, pysol_update_array, sol_generator


@pysol
def hello_world():
    # ── kernel imports ────────────────────────────────────────────────────────
    hello_world_k = import_kernel("hello_world_k", [])

    # ── body ──────────────────────────────────────────────────────────────────
    hello_world_k()
    vi : Type.Int = 1234
    vb : Type.Float = vi + 1
    n : Type.Int= 10
    print_k = import_kernel("print", [KernelArgument("value", True, Type.Int)])
    print_k(vi)
    A = Array(1, 1, [n], [20], Type.Float)
    B = Array(1, 1, [n], [20], Type.Float)
    C = Array(1, 1, [n], [20], Type.Float)
    init_k = import_kernel("init_vec", [KernelArgument("i", True, Type.Int), KernelArgument("A", False, Type.Float, 1)])
    for i in range(n):
        init_k(i, A[0, i])
        init_k(i * 2, B[0, i])
    print_vec_k = import_kernel("print_vec", [KernelArgument("i", True, Type.Int), KernelArgument("A", True, Type.Float, 1)])
    for i in range(n):
        print_vec_k(i, A[0, i])
        print_vec_k(i, B[0, i])
    C = A + B
    for i in range(n):
        print_vec_k(i, C[0, i])
    D = Array(2, 2, [n, n], [n + 2, n + 2], Type.Float)
    E = Array(2, 2, [n, n], [n + 2, n + 2], Type.Float)
    pysol_update_array(D, 0, E, 0)
    it : Type.Int = 1
    while it < 100:
        it = it + 1
    print_k(it)

@pysol
def solver():
    init_el_k = import_kernel("init_el",
                              [KernelArgument("ii", True, Type.Int),
                               KernelArgument("jj", True, Type.Int),
                               KernelArgument("n", True, Type.Int),
                               KernelArgument("m", True, Type.Int),
                               KernelArgument("A", False, Type.Double, 2)])
    print_el_k = import_kernel("print_el",
                               [KernelArgument("ii", True, Type.Int),
                               KernelArgument("jj", True, Type.Int),
                               KernelArgument("A", False, Type.Double, 2)])
    calc_main_k = import_kernel("calc_main",
                                [KernelArgument("a", True, Type.Double),
                                 KernelArgument("dt", True, Type.Double),
                                 KernelArgument("dx", True, Type.Double),
                                 KernelArgument("dy", True, Type.Double),
                                 KernelArgument("prev", True, Type.Double, 2),
                                 KernelArgument("curr", False, Type.Double, 2)], [pysol.CPU, pysol.GOOPAX])
    print_it_k = import_kernel("print_iter", [KernelArgument("it", True, Type.Int)])
    nb : Type.Int = 10
    mb : Type.Int = 10
    block_n_ : Type.Int = 100
    block_m_ : Type.Int = 100

    a : Type.Double = 100
    dx : Type.Double = 0.01
    dy : Type.Double = 0.01
    dx2 : Type.Double = dx * dx
    dy2 : Type.Double = dy * dy
    dt : Type.Double = dx2 * dy2 / (2.0 * a * (dx2 + dy2))

    A = Array(2, 2, [nb, nb], [block_n_ + 2, block_m_ + 2], Type.Double)
    for i in range(nb):
        for j in range(mb):
            init_el_k(i, j, nb, mb, A[i][j])

    it : Type.Int = 0
    while it < 10000:
        print_it_k(it)
        for i in range(nb):
            for j in range(mb):
                calc_main_k(a, dt, dx, dy, A[it, i][j], A[it + 1, i][j])
        pysol_update_array(A, it + 1, A, it + 2)
        it = it + 2
    for i in range(nb):
        for j in range(mb):
            print_el_k(i, j, A[it, i][j])

def print_command_line_arguments():
    for arg in sys.argv[1:]:
        print(arg)


if __name__ == "__main__":
    print_command_line_arguments()
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
    pysol_compile("solver")
    pysol_run("solver")
