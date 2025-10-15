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
jl.Pkg.add(url="http://github.com/RomeoV/RunwayLib.jl", rev="docs")  # should land on master asap
```
You only need to do this once. Now you should be able to use `RunwayLib` from python like so:
```python
{{PYTHON_EXAMPLE}}
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
