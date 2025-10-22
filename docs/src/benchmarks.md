# Benchmarks
```@raw html
<style>
@media screen and (min-width: 1056px) {
  #documenter .docs-main {
    max-width: none !important;
  }
  #documenter .docs-main pre {
    max-width: 52rem !important;
  }
  #documenter .docs-main p,
  #documenter .docs-main ul,
  #documenter .docs-main ol,
  #documenter .docs-main h1,
  #documenter .docs-main h2,
  #documenter .docs-main h3,
  #documenter .docs-main h4,
  #documenter .docs-main h5,
  #documenter .docs-main h6 {
    max-width: 52rem !important;
  }
}
</style>
```
Here we present a benchmarking example of the simple estimation presented in [Getting Started](@ref).
In particular, this is an example of how we can make use of statically sized arrays via StaticArrays.jl for wide parts of the code and optimization, letting us eliminate most allocations.
Below in [Profiling the Callstack](@ref) we will go into further details.
Here, we further make use of pre-allocated caches for various steps of the optimization such as the Jacobians, and algorithmic magic such as a well-tuned implementation of [`LevenbergMarquardt`](@extref :jl:function:`NonlinearSolveFirstOrder.LevenbergMarquardt`) with [`GeodesicAcceleration`](@extref :jl:type:`NonlinearSolveBase.GeodesicAcceleration`).

```@example benchmarkrun
using RunwayLib, Unitful.DefaultSymbols, Rotations, StaticArrays
using BenchmarkTools
runway_corners = SA[
    WorldPoint(0.0m, 50m, 0m),      # near left
    WorldPoint(3000.0m, 50m, 0m),   # far left
    WorldPoint(3000.0m, -50m, 0m),  # far right
    WorldPoint(0.0m, -50m, 0m),     # near right
]

cam_pos = WorldPoint(-2000.0m, 12m, 150m)
cam_rot = RotZYX(roll=1.5°, pitch=5°, yaw=0°)

true_observations = [project(cam_pos, cam_rot, p) for p in runway_corners] |> SVector
noisy_observations = [p + ProjectionPoint(2.0*randn(2)px) for p in true_observations] |> SVector

@benchmark estimatepose6dof(PointFeatures(runway_corners, noisy_observations))
```
At the time of writing[^1] the median benchmarking time for the pose estimate measures around `228.088 μs` (about `0.2` milliseconds) for one pose estimation. *Quite fast!*
We also see that there is some spread, with typically up to 50% slower execution (`0.3` milliseconds), with very rare extreme outliers taking way longer -- about `86` milliseconds at the time of writing.
The cause of the rare outlier is still under investigation, and we assume interference of the CPU on the GitHub runner generating these results.
Nonetheless, we conclude that we can estimate the pose from noisy point predictions extremely efficiently -- typically about five thousand times per second.

## Profiling the Callstack
To further highlight the efficiency of the presented code we can investigate the profiling trace of the call graph.
Here we present a visualization thereof, although it is necessary to **scroll down quite far to skip past the call stack of all the various "entrypoints"**.

```@example benchmarkrun
using ProfileCanvas
# once for precompile
@profview map(1:1000) do _; estimatepose6dof(PointFeatures(runway_corners, noisy_observations)); end
# now we measure
@profview map(1:1000) do _; estimatepose6dof(PointFeatures(runway_corners, noisy_observations)); end
```

```@raw html
<br>
```

We can again see that we have gotten rid of almost all allocations (visualized in orange), except for `pose_optimization_objective`, which is currently a restriction of `NonlinearSolve.jl`.
Nonetheless, we highlight that we spend almost the entire rest of the duration on actual compute rather than kernel calls such as allocations.


[^1]: The outputs here are generated for every build.
