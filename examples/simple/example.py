"""
Simple pysol example.

Demonstrates:
  - 1-D array declarations (float and double blocks)
  - Two kernel imports with scalar and array-element parameters
  - A for-loop over array elements
  - A nested if-statement using a comparison condition
  - Array element access via .get()

Run from the project root:
    python examples/simple/example.py

The decorator writes  process_data.sol  next to this file at import time.
The __main__ block calls the function (showing the live analysis) and then
prints the generated .sol source.
"""

import os
from pysol import Array, KernelArgument, Type, import_kernel, pysol


@pysol
def process_data():
    # ── array declarations ────────────────────────────────────────────────────
    input_arr  = Array(1, 1, [64], [32], Type.Float)
    output_arr = Array(1, 1, [64], [32], Type.Double)

    # ── kernel imports ────────────────────────────────────────────────────────
    transform = import_kernel("transform", [
        KernelArgument("count", True,  Type.Int),
        KernelArgument("src",   True,  Type.Float,  n_dims=1),
        KernelArgument("dst",   False, Type.Double, n_dims=1),
    ])
    normalize = import_kernel("normalize", [
        KernelArgument("src",    True,  Type.Double, n_dims=1),
        KernelArgument("result", False, Type.Double, n_dims=1),
    ])

    # ── body ──────────────────────────────────────────────────────────────────
    for i in range(64):
        transform(i, input_arr.get(0, [i]), output_arr.get(0, [i]))
        if i < 32:
            normalize(output_arr.get(0, [i]), output_arr.get(0, [i]))


if __name__ == "__main__":
    # Call the function — the decorator prints the analysis summary first.
    process_data()

    # Print the generated .sol file.
    sol_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "process_data.sol"
    )
    print("\n" + "=" * 60)
    print(f"Generated: {sol_path}")
    print("=" * 60)
    with open(sol_path) as fh:
        print(fh.read())
