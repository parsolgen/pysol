
import os
import sys
from pysol import GrowingArray, KernelArgument, Type, import_kernel, pysol, pysol_compile, pysol_run, load_config, pysol_struct, sol_generator

# 3D grid of 4x4x4 cells, each a GrowingArray<Sample> cell (a dynamically
# growing vector of Sample structs). Kept small on purpose — this is a
# correctness check for @pysol_struct + GrowingArray, not a solver.
NB = 4
MB = 4
LB = 4


# ── struct type ──────────────────────────────────────────────────────────────
# A @pysol_struct class is declared like @dataclass: every field must be
# Type.Int, Type.Float, or Type.Double. It can then be used as the element
# type of an Array / GrowingArray, or as a variable's type annotation.
@pysol_struct
class Sample:
    a: Type.Int
    b: Type.Float
    c: Type.Double


@pysol
def growing_array_struct_test():
    # ── kernel imports ────────────────────────────────────────────────────────
    hello_world_k = import_kernel("hello_world_k", [])

    init_k = import_kernel("init_struct_cell3d",
                            [KernelArgument("ii", True, Type.Int),
                             KernelArgument("jj", True, Type.Int),
                             KernelArgument("kk", True, Type.Int),
                             KernelArgument("s", False, Sample)])
    check_k = import_kernel("check_struct_cell3d",
                             [KernelArgument("ii", True, Type.Int),
                              KernelArgument("jj", True, Type.Int),
                              KernelArgument("kk", True, Type.Int),
                              KernelArgument("s", True, Sample)])

    # ── body ──────────────────────────────────────────────────────────────────
    hello_world_k()

    nb : Type.Int = 4
    mb : Type.Int = 4
    lb : Type.Int = 4

    A = GrowingArray(3, [nb, mb, lb], Sample)

    for i in range(nb):
        for j in range(mb):
            for k in range(lb):
                init_k(i, j, k, A[i][j][k])

    for i in range(nb):
        for j in range(mb):
            for k in range(lb):
                check_k(i, j, k, A[i][j][k])


def verify():
    """Read the per-cell results written by check_struct_cell3d and assert
    that every GrowingArray<Sample> cell holds exactly the expected value."""
    failures = []
    for i in range(NB):
        for j in range(MB):
            for k in range(LB):
                path = f"struct3d_{i}_{j}_{k}"
                with open(path) as fh:
                    mismatches = int(fh.read().strip())
                if mismatches:
                    failures.append((i, j, k, mismatches))

    if failures:
        print(f"FAIL: {len(failures)}/{NB * MB * LB} cell(s) with mismatches: {failures}")
        sys.exit(1)

    print(f"PASS: GrowingArray<Sample> values verified across {NB * MB * LB} cells")


if __name__ == "__main__":
    load_config("config.json")

    print("==========GROWING ARRAY STRUCT TEST============")
    pysol_compile("growing_array_struct_test")
    pysol_run("growing_array_struct_test")
    verify()
