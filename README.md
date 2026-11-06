# pysol

pysol is a Python DSL and compiler front-end for **ParSolGen**. You write ordinary-looking
Python functions using a small, restricted subset of the language plus a handful of pysol
primitives (`Array`, `GrowingArray`, `CSRMatrix`, `import_kernel`, ...); the `@pysol` decorator
statically analyses the function's AST (no execution) and emits a `.sol` source file. ParSolGen's
`sgc` compiler then turns that `.sol` file into C++ which you build into a native shared library
and run from Python.

## Features

- **`@pysol`** — decorate a function to statically analyse it (arrays, loops, `if` statements,
  kernel imports) and generate a `<function_name>.sol` file next to the source.
- **`Array` / `GrowingArray` / `CSRMatrix`** — typed, multi-dimensional, blocked array
  descriptors, including explicit version dimensions for time-stepping / stencil codes.
- **`import_kernel`** — declare an externally-implemented kernel (a plain C++ function) with a
  typed argument list, callable from inside a `@pysol` function. Each kernel can declare which
  backends it is implemented for via `supported_backends_list`:

  ```python
  from pysol import CPU, GOOPAX, KernelArgument, Type, import_kernel

  calc_k = import_kernel(
      "calc_main",
      [KernelArgument("a", True, Type.Double), KernelArgument("out", False, Type.Double)],
      [CPU, GOOPAX],   # defaults to [CPU] when omitted
  )
  ```

- **`pysol_struct`** — define small POD struct types shared between Python and the generated
  kernels.
- **`pysol_update_array` / `pysol_update_array_3d_26_full_sum`** — generate the ghost/halo
  exchange patterns used by stencil and particle codes.
- **`pysol_compile` / `pysol_run`** — drive ParSolGen's `sgc` compiler and your C++ compiler to
  turn the generated `.sol` file into a `.so`, then load and run it.
- **`load_config` / `get_config`** — load the paths and compiler flags pysol needs from a
  per-project `config.json`.

## Requirements

- Python >= 3.8
- **ParSolGen** — pysol only *generates* `.sol` files and drives the build; it does not vendor or
  build ParSolGen itself. You need a working ParSolGen installation available on the machine you
  compile/run on:
  - the `sgc` compiler binary (turns `.sol` → `.cpp`)
  - the ParSolGen C++ runtime headers and `libPSRts` (linked into the generated `.so`)
  - a C++ compiler (e.g. `mpicxx`) capable of building the generated code
  - any extra libraries the kernels you compile against pull in (e.g. MKL, TBB, GKlib, GOOPAX —
    only needed if you actually target the `GOOPAX` backend)

  pysol has no way to detect or install ParSolGen for you — get the paths right in `config.json`
  (see below) before calling `pysol_compile`.

## Installation

From the repository root:

```bash
pip install .            # regular install
pip install -e ".[dev]"  # editable install + pytest, for development
```

Metadata lives in `setup.cfg`/`setup.py` rather than `pyproject.toml`'s `[project]` table, so this
works with both old and modern `setuptools`.

## Quick example

```python
from pysol import Array, KernelArgument, Type, import_kernel, pysol, pysol_compile, pysol_run, load_config

@pysol
def hello_world():
    print_k = import_kernel("print", [KernelArgument("value", True, Type.Int)])
    n : Type.Int = 10
    A = Array(1, 1, [n], [20], Type.Float)
    init_k = import_kernel("init_vec", [KernelArgument("i", True, Type.Int),
                                         KernelArgument("A", False, Type.Float, 1)])
    for i in range(n):
        init_k(i, A[0, i])
    print_k(n)

if __name__ == "__main__":
    load_config("config.json")   # points pysol at sgc, your compiler, and link flags
    pysol_compile("hello_world") # .sol -> .cpp -> hello_world.so (via sgc + your C++ compiler)
    pysol_run("hello_world")     # load hello_world.so and run it
```

See `examples/simple/hello.py` for a fuller version, and `examples/` in general for larger
ported examples (ray tracers, a PIC3D plasma solver, cellular-automaton gas models, ...).

## `config.json`

`load_config(...)` reads a JSON object with the paths/flags `pysol_compile` needs to drive
ParSolGen and your C++ compiler, e.g.:

```json
{
    "sgc": "/path/to/ParSolGen/build/src/Parser/sgc",
    "compiler": "mpicxx",
    "compile_options": "-std=c++23 -I/path/to/ParSolGen/include -L/path/to/ParSolGen/lib -lPSRts -fopenmp -O3"
}
```

Adjust the paths and flags to match where ParSolGen and its dependencies are installed on your
machine; `compile_options` may be a single string (split on whitespace) or a list of strings.

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

## Project layout

- `src/pysol/` — the library itself (AST analysis, `.sol` generation, array/kernel primitives).
- `examples/` — ported example programs, each with its own `config.json`.
- `tests/` — unit tests (`pytest`).
