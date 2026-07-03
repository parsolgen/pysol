
import os
import sys
from pysol import Array, KernelArgument, Type, import_kernel, pysol, pysol_compile, pysol_run, load_config


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
    D = Array(1, 1, [n], [20], Type.Float)
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
    E = Array(2, 2, [n, n], [10, 10], Type.Float)


def print_command_line_arguments():
    for arg in sys.argv[1:]:
        print(arg)


if __name__ == "__main__":
    print_command_line_arguments()
    load_config("config.json")
    # Call the function — the decorator prints the analysis summary first.
    hello_world()

    # Print the generated .sol file.
    sol_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "hello_world.sol"
    )
    print("\n" + "=" * 60)
    print(f"Generated: {sol_path}")
    print("=" * 60)
    if not os.path.exists("hello_world.so"):
        pysol_compile("hello_world")
    pysol_run("hello_world")
    with open(sol_path) as fh:
        print(fh.read())
