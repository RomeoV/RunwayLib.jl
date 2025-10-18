# C Interface

Beyond the Julia and Python interfaces, we also expose core parts of our library as a ahead-of-time compiled C library using [`JuliaC.jl`](https://github.com/JuliaLang/JuliaC.jl).
This is currently in a proof-of-concept state since not all public functions are exported, but it will not be hard to expand this once the need is there.

## Usage example
A usage example can be found at 
[`main.c`](https://github.com/RomeoV/RunwayLib.jl/blob/master/juliac/main.c)
which we reprint here:
````@eval
import Markdown
Markdown.parse("""
```C
$(readchomp(joinpath("..", "..", "juliac", "main.c")))
```
""")
````

Next to the `main.c` file one can find the [`Makefile`](https://github.com/RomeoV/RunwayLib.jl/blob/master/juliac/main.c) which outlines how to compile using the generated C library.


!!! warning
    At this time, only `estimate_pose_6dof` is properly supported, although there's nothing stopping us from supporting the rest.
