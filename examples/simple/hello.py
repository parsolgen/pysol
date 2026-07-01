
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
    print_k = import_kernel("print", [KernelArgument("value", True, Type.Int)])
    print_k(vi)


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
    pysol_compile("hello_world")
    pysol_run("hello_world")
    with open(sol_path) as fh:
        print(fh.read())
