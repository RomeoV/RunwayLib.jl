# Getting Started

*TLDR*
```@example gettingstarted
using RunwayLib, Unitful.DefaultSymbols, Rotations
import Rotations: params

runway_corners = [
    WorldPoint(0.0m, 50m, 0m),     # near left
    WorldPoint(3000.0m, 50m, 0m),  # far left
    WorldPoint(3000.0m, 50m, 0m),  # far right
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
