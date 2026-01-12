## Core design principles
- try to use Julia dispatch mechanism to create flexible yet hard-to-misuse interfaces
  + for instance, we have good typing on fundametal types such as `WorldPoint` that represents a 3d point in world coordinates
  + however, we allow parametrized `eltype`s that allow e.g. for Dual numbers
- core code is "unitful", but it's not a strict requirement
  + initially we tried to make both interfaces and internal code "unitful", i.e., be explicit about floats having units such as "meters", "pixels", etc.
  + we still want to generally maintain this structure. However, for now we can post-pone it to a "post-processing step" after we have made sure the actual code works. Nonetheless, where it exists it should be kept.
  + having to add or remove units manually is an anti-pattern and should be used sparingly, but is necessary for some operations (e.g. interaction with NonlinearSolve.jl)
  + because unitful errors are hard to read, when writing a new function it's ok to first ustrip, make the code work, and then add units iteratively during development
- trimmability
  + the library aims to support a use case as a C library. for this we use the  JuliaC library which traces the compilation starting at some "entrypoints"
  + trimming is a recent and somewhat fragile feature. essentially, the question whether the code can be trimmed is equivalent to the question whether the code is
    1. type-stable (function input types determine output types)
    2. type-grounded (every variable can have its type inferred based on function input types)
  + testing for trimmability is hard, but can be done with `JET.@report_opt` (or `test_opt`). however, errors are hard to read, so this is useful to be applied at a precise level, and generally just reasoning about code type inference can help
- static arrays
  + to maximize performance, we try to support optional use of static arrays everywhere. this essentialy boils down to trying to match the array types of the inputs, which many julia functions support. sometimes we need something like `similar` etc, but we should try to refrain from explicit arrays such as `zeros` etc.
  + if this is not always possible that's fine, we can always iteratively improve on this later
  + most functions should have a test making sure that if we put static arrays in, we get static arrays out. nothing too fancy

## Package Structure
- This package has three main components. The Julia code (`src`), the compiled library (`juliac`), and some python bindings code
- Here's the most important directory `tree`
```
romeo@Romeo-P1 ~/D/S/A/RunwayLib (seed-other-test-rngs) [1]> tree -L 2
.
├── CHANGELOG.md
├── CLAUDE.md
├── docs
│   ├── build
│   ├── CondaPkg.toml
│   ├── juliapkg.json
│   ├── make.jl
│   ├── Manifest.toml
│   ├── Project.toml
│   ├── src
│   └── test.md
├── ext
│   └── PythonCallExt.jl
├── foo.svg
├── input.typ
├── juliac
│   ├── juliac
│   ├── juliac-script.jl
│   ├── libposeest.h
│   ├── loadrunwaylib.jl
│   ├── main.c
│   ├── mainc
│   ├── mainc_c99
│   ├── main.jl
│   ├── Makefile
│   ├── Project.toml
│   ├── RunwayLibCompiled
│   └── test_in_container.sh
├── main.py
├── Manifest.toml
├── Manifest-v1.12.toml
├── Project.toml
├── pyproject.toml
├── README.md
├── src
│   ├── camera_model
│   ├── c_api.jl
│   ├── coordinate_systems
│   ├── data_management
│   ├── entrypoints.jl
│   ├── integrity
│   ├── piracy.jl
│   ├── pose_estimation
│   ├── precompile_workloads.jl
│   ├── RunwayLib.jl
│   ├── sum_type_example.jl
│   └── uncertainty_quantification
├── test
│   ├── c_interface
│   ├── integration
│   ├── Manifest.toml
│   ├── Manifest-v1.12.toml
│   ├── Project.toml
│   ├── python_interface
│   ├── runtests.jl
│   └── unit
├── test_linearsolve_regression
│   ├── Manifest.toml
│   ├── Project.toml
│   └── test_lm.jl
```

## Testing Tips

- to avoid the TTFX overhead of Julia we're trying to rely on the julia-repl mcp. you should for instance be able to run test files via an "include" statement. You can assume that the project's "test" project is activated. You can check via `Pkg.status()` and warn if you need another project.

