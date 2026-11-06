
import os
import sys
from pysol import Array, KernelArgument, Type, import_kernel, pysol, pysol_compile, pysol_run, load_config, pysol_update_array, sol_generator

# Block grid: 10x10x10 blocks, each block a 10x10x10 cube of Doubles.
# init_el_3d fills every cell of block (ii, jj, kk) uniformly with ii*jj*kk,
# so sum()/max()/min() over the whole array can be computed independently in
# Python and compared against what the generated reduce() calls produce.
# NB/MB/LB/BLOCK_* must stay in sync with the nb/mb/lb/block_*_ literals
# inside reduce_array_3d_test() below — the @pysol AST analysis only
# understands literals local to the decorated function.
NB = 10
MB = 10
LB = 10
BLOCK_N = 10
BLOCK_M = 10
BLOCK_L = 10


@pysol
def reduce_array_3d_test():
    # ── kernel imports ────────────────────────────────────────────────────────
    hello_world_k = import_kernel("hello_world_k", [])
    print_value_k = import_kernel("printv", [KernelArgument("v", True, Type.Double)])
    write_sum_k = import_kernel("write_sum_result", [KernelArgument("v", True, Type.Double)])
    write_max_k = import_kernel("write_max_result", [KernelArgument("v", True, Type.Double)])
    write_min_k = import_kernel("write_min_result", [KernelArgument("v", True, Type.Double)])

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

    sum_value : Type.Double = sum(A)
    max_value : Type.Double = max(A)
    min_value : Type.Double = min(A)

    print_value_k(sum_value)
    print_value_k(max_value)
    print_value_k(min_value)

    write_sum_k(sum_value)
    write_max_k(max_value)
    write_min_k(min_value)


def verify():
    """Compare the sum()/max()/min() reduction results (written to files by
    write_*_result) against values computed independently in Python."""
    cells_per_block = BLOCK_N * BLOCK_M * BLOCK_L
    coord_sum = sum(ii * jj * kk for ii in range(NB) for jj in range(MB) for kk in range(LB))

    expected = {
        "sum": cells_per_block * coord_sum,
        "max": (NB - 1) * (MB - 1) * (LB - 1),
        "min": 0,
    }

    failures = []
    for op_name, exp_value in expected.items():
        path = f"reduce_{op_name}_result"
        with open(path) as fh:
            actual = float(fh.read().strip())
        if abs(actual - exp_value) > 1e-6:
            failures.append((op_name, exp_value, actual))

    if failures:
        for op_name, exp_value, actual in failures:
            print(f"FAIL: {op_name}() expected {exp_value}, got {actual}")
        print("FAIL")
        sys.exit(1)

    print(
        f"PASS: sum()={expected['sum']}, max()={expected['max']}, "
        f"min()={expected['min']} all verified"
    )


if __name__ == "__main__":
    load_config("config.json")

    print("==========REDUCE ARRAY 3D TEST============")
    pysol_compile("reduce_array_3d_test")
    pysol_run("reduce_array_3d_test")
    verify()
