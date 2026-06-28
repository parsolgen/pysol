
import os
from pysol import Array, KernelArgument, Type, import_kernel, pysol


@pysol
def hello_world():
    # ── kernel imports ────────────────────────────────────────────────────────
    hello_world_k = import_kernel("hello_world", [])

    # ── body ──────────────────────────────────────────────────────────────────
    hello_world_k()

if __name__ == "__main__":
    # Call the function — the decorator prints the analysis summary first.
    hello_world()

    # Print the generated .sol file.
    sol_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "hello_world.sol"
    )
    print("\n" + "=" * 60)
    print(f"Generated: {sol_path}")
    print("=" * 60)
    with open(sol_path) as fh:
        print(fh.read())
