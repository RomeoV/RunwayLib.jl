# Getting Started

*TLDR*
```jldoctest
using RunwayLib, Unitful.DefaultSymbols, Rotations
import Rotations: params

world_points = [
    WorldPoint(0.0m, 50m, 0m),
    WorldPoint(3000.0m, 50m, 0m),
    WorldPoint(3000.0m, 50m, 0m),
    WorldPoint(0.0m, -50m, 0m),
]

cam_pos = WorldPoint(-2000.0m, 12m, 150m)
cam_rot = RotZYX(roll=1.5°, pitch=5°, yaw=0°)

true_observations = [project(cam_pos, cam_rot, p) for p in world_points]
noisy_observations = [p + ProjectionPoint(1.0px*randn(2)) for p in true_observations]

(cam_pos_est, cam_rot_est) = estimatepose6dof(
    PointFeatures(world_points, noisy_observations)
)[(:pos, :rot)]
(y, p, r) = params(cam_rot_est)

using Test
@test cam_pos ≈ cam_pos_est rtol=0.01
@test params(cam_rot) ≈ params(cam_rot_est) rtol=0.1
true

# output

true
```


