# Python Interface
Several attempts have been made to make this module easily callable from python.
At this time, we recommend using the python package [JuliaCall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/).
Simply install juliacall alongside numpy using your favorite method, for instance
```bash
uv add juliacall
uv add numpy
```
and use juliacall to install `RunwayLib` and the `PythonCall` extension which enables
the python API.
```python
from juliacall import Main as jl
jl.Pkg.add(url="http://github.com/RomeoV/RunwayLib.jl")  # will be registered soon
```
You only need to do this once. Now you should be able to use `RunwayLib` from python like so:
```python
import numpy as np
from juliacall import Main as jl
jl.seval("using RunwayLib, PythonCall")
jl.seval("import RunwayLib: px, m")

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
pos = np.asarray(jl.broadcast(jl.ustrip, res_jl.pos))  # or `np.array(..., copy=None)`
rot = np.asarray(res_jl.rot)

for i in range(1_000):
    jl.estimatepose6dof(points3d, points2d, camconf)

# Smoke test assertions
assert pos.shape == (3,), f"Expected position shape (3,), got {pos.shape}"
assert rot.shape == (3, 3), f"Expected rotation shape (3, 3), got {rot.shape}"
print(f"✓ Smoke test passed! Position: {pos}, Rotation shape: {rot.shape}")

```

Notice that we can also directly wrap `np.array` for `WorldPoint` and the other, e.g.:
```python
points2d_np = [
    np.array([2284.8, 619.32]),
    np.array([2355.2, 620.90]),
    np.array([2399.5, 903.30]),
    np.array([2252.4, 904.09]),
]
point2d = [jl.ProjectionPoint(p) for p in points2d_np]
```
