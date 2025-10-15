# Python Interface
Several attemps have been made to make this module easily callable from python.
At this time, we recommend using the python package [JuliaCall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/).
Simply install the package using your favorite method, for instance
```python
uv add juliacall
```
and use juliacall to install `RunwayLib` and the `PythonCall` extension which enables
the python API.
```python
from juliacall import Main as jl
jl.Pkg.develop(url="http://github.com/RomeoV/RunwayLib.jl#docs")
jl.Pkg.add("PythonCall")
```
You only need to do this once. Now you should be able to use `RunwayLib` like so:
```python
import numpy as np
from juliacall import Main as jl
jl.seval("using RunwayLib, PythonCall")
jl.seval("import RunwayLib: m, px")

points2d = [
    jl.ProjectionPoint(2284.8, 619.32)*jl.px,
    jl.ProjectionPoint(2355.2, 620.90)*jl.px,
    jl.ProjectionPoint(2399.5, 903.30)*jl.px,
    jl.ProjectionPoint(2252.4, 904.09)*jl.px,
]
points3d = [
    jl.WorldPoint(1600, 15, -8)*jl.m,
    jl.WorldPoint(1600, -15, -8)*jl.m,
    jl.WorldPoint(0, -15, 0)*jl.m,
    jl.WorldPoint(0, 15, 0)*jl.m,
]
intrinsic_matrix = np.array(
    [
        [-7311.8, 0.0, 2032.3],
        [0.0, -7311.8, 1707.4],
        [0.0, 0.0, 1.0],
    ]
)

camconf = jl.CameraMatrix[jl.Symbol("offset")](
    intrinsic_matrix, jl.px(2048.0), jl.px(1024.0),
)

jl.seval("using RunwayLib.Unitful: ustrip")
res_jl = jl.estimatepose6dof(points3d, points2d, camconf)
pos = np.array(jl.broadcast(jl.ustrip, res_jl.pos))
rot = np.array(res_jl.rot)

for i in range(1_000):
    jl.estimatepose6dof(points3d, points2d, camconf)
```
