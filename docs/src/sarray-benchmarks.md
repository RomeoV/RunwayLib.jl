# StaticArray Benchmarks

We explore the compute time requirements for pose estimation under different configurations:
comparing **3dof vs 6dof**, the impact of **StaticArrays**, and solving with **additional line angle features**.

## Setup

```@example sarray_bench
using RunwayLib, Unitful.DefaultSymbols, Rotations
using StaticArrays, Chairmarks, Printf, Markdown

runway_corners = [
    WorldPoint(0.0m, 50m, 0m),     # near left
    WorldPoint(3000.0m, 50m, 0m),  # far left
    WorldPoint(3000.0m, -50m, 0m),  # far right
    WorldPoint(0.0m, -50m, 0m),    # near right
]

cam_pos = WorldPoint(-2000.0m, 12m, 150m)
cam_rot = RotZYX(roll=1.5°, pitch=5°, yaw=0°)

true_observations = [project(cam_pos, cam_rot, p) for p in runway_corners]
noisy_observations = [p + ProjectionPoint(2.0*randn(2)px) for p in true_observations]

fmt(t) = @sprintf("%.1f μs", t * 1e6)
speedup(a, b) = @sprintf("%.2f×", a / b)

pf_arr = PointFeatures(runway_corners, noisy_observations)
nothing # hide
```

## 3dof vs 6dof

We first compare the two solver modes using regular `Vector`-backed `PointFeatures`.

```@example sarray_bench
b_6dof = minimum(@be pf_arr estimatepose6dof)
b_3dof = minimum(@be pf_arr estimatepose3dof(_, NO_LINES, cam_rot))

Markdown.parse("""
|                | 6dof            | 3dof            | Speedup          |
|:---------------|:----------------|:----------------|:-----------------|
| Regular arrays | $(fmt(b_6dof.time)) | $(fmt(b_3dof.time)) | $(speedup(b_6dof.time, b_3dof.time)) |
""")
```

## Impact of StaticArrays

Wrapping the point features in `SVector` allows the compiler to unroll loops and avoid heap allocations.

```@example sarray_bench
pf_sa = PointFeatures(SVector{4}(runway_corners), SVector{4}(noisy_observations))

b_6dof_sa = minimum(@be pf_sa estimatepose6dof)
b_3dof_sa = minimum(@be pf_sa estimatepose3dof(_, NO_LINES, cam_rot))

Markdown.parse("""
|                  | 6dof            | 3dof            | Speedup          |
|:-----------------|:----------------|:----------------|:-----------------|
| Regular arrays   | $(fmt(b_6dof.time)) | $(fmt(b_3dof.time)) | $(speedup(b_6dof.time, b_3dof.time)) |
| StaticArrays     | $(fmt(b_6dof_sa.time)) | $(fmt(b_3dof_sa.time)) | $(speedup(b_6dof_sa.time, b_3dof_sa.time)) |
| SA speedup       | $(speedup(b_6dof.time, b_6dof_sa.time)) | $(speedup(b_3dof.time, b_3dof_sa.time)) |   |
""")
```

## With line angle features

Adding observed runway edge angles as `LineFeatures` provides additional constraints to the solver.

```@example sarray_bench
line_pts = [
    (runway_corners[1], runway_corners[2]),
    (runway_corners[3], runway_corners[4]),
]
true_lines = map(line_pts) do (p1, p2)
    proj1 = project(cam_pos, cam_rot, p1)
    proj2 = project(cam_pos, cam_rot, p2)
    getline(proj1, proj2)
end
observed_lines = [
  Line(r + 1px*randn(), theta + deg2rad(1°)*randn())
  for (; r, theta) in true_lines
]

lf_arr = LineFeatures(line_pts, observed_lines)
lf_sa  = LineFeatures(SVector{2}(line_pts), SVector{2}(observed_lines))

bl_6dof    = minimum(@be (pf_arr, lf_arr) estimatepose6dof(_...))
bl_3dof    = minimum(@be (pf_arr, lf_arr) estimatepose3dof(_..., cam_rot))
bl_6dof_sa = minimum(@be (pf_sa,  lf_sa)  estimatepose6dof(_...))
bl_3dof_sa = minimum(@be (pf_sa,  lf_sa)  estimatepose3dof(_..., cam_rot))

Markdown.parse("""
|                  | 6dof            | 3dof            | Speedup          |
|:-----------------|:----------------|:----------------|:-----------------|
| Regular arrays   | $(fmt(bl_6dof.time)) | $(fmt(bl_3dof.time)) | $(speedup(bl_6dof.time, bl_3dof.time)) |
| StaticArrays     | $(fmt(bl_6dof_sa.time)) | $(fmt(bl_3dof_sa.time)) | $(speedup(bl_6dof_sa.time, bl_3dof_sa.time)) |
| SA speedup       | $(speedup(bl_6dof.time, bl_6dof_sa.time)) | $(speedup(bl_3dof.time, bl_3dof_sa.time)) |   |
""")
```
