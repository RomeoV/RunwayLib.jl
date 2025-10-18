# RunwayLib.jl

## Getting Started

```@example gettingstarted
using RunwayLib, Unitful.DefaultSymbols, Rotations

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

(cam_pos_est, cam_rot_est) = estimatepose6dof(
    PointFeatures(runway_corners, noisy_observations)
)[(:pos, :rot)]

cam_pos_est
```

We can extract roll-pitch-yaw as
```@example gettingstarted
import Rotations: params
(yaw, pitch, roll) = params(cam_rot_est)
@show rad2deg(roll*rad)
@show rad2deg(pitch*rad)
@show rad2deg(yaw*rad)
;
```

## Using Line Features
Besides point features we can additionally include line features which can typically improve
our altitude and crosstrack estimations, but usually can't improve our alongtrack estimation much because the line projections are constant along the glidepath.
See [Line Projections](@ref) for more information on the line parameterization.

```@example gettingstarted
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
  Line(
    r + 1px*randn(),
    theta + deg2rad(1°)*randn()
  )
  for (; r, theta) in true_lines
]

# now with additional line features
(cam_pos_est, cam_rot_est) = estimatepose6dof(
    PointFeatures(runway_corners, noisy_observations),
    LineFeatures(line_pts, observed_lines)
)[(:pos, :rot)]

cam_pos_est
```
