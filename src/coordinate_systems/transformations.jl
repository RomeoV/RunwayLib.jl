"""
Coordinate system transformations for runway pose estimation.

This module provides functions to transform points between different coordinate systems:
- World to camera coordinates
- Camera to world coordinates
- Camera to image projection coordinates
"""

using LinearAlgebra
using Rotations
using Unitful

"""
    world_pt_to_cam_pt(cam_pos::WorldPoint, R::RotZYX, world_pt::WorldPoint) -> CameraPoint

Transform a point from world coordinates to camera coordinates.

# Arguments
- `cam_pos::WorldPoint`: Camera position in world coordinates
- `cam_rot::RotZYX`: Camera orientation (ZYX Euler angles: yaw, pitch, roll)
- `world_pt::WorldPoint`: Point to transform in world coordinates

# Returns
- `CameraPoint`: Point in camera coordinate system

# Algorithm
1. Translate point relative to camera position
2. Rotate by inverse of camera rotation to get camera-relative coordinates

# Examples
```jldoctest world_pt_to_cam_pt_doctest
using RunwayLib, Unitful.DefaultSymbols, Rotations
# Camera at origin with no rotation
cam_pos = WorldPoint(0.0m, 0m, 0m)
cam_rot = RotZYX(roll=0.0rad, pitch=0rad, yaw=0rad)  # No rotation
world_pt = WorldPoint(1.0m, 2m, 3m)

cam_pt = world_pt_to_cam_pt(cam_pos, cam_rot, world_pt)

# output

3-element WorldPoint{Float64{m}} with indices SOneTo(3):
 1.0 m
 2.0 m
 3.0 m
```

With some rotation:

```jldoctest world_pt_to_cam_pt_doctest
# Camera with 90-degree yaw rotation
cam_rot = RotZYX(yaw=π/2rad, roll=0.0rad, pitch=0.0rad)  # 90-degree yaw
world_pt = WorldPoint(1.0m, 0.0m, 0.0m)

cam_pt = world_pt_to_cam_pt(cam_pos, cam_rot, world_pt)
# After rotation, x becomes -y in camera frame
cam_pt ≈ WorldPoint(0.0m, -1.0m, 0.0m)

# output

true
```
"""
function world_pt_to_cam_pt(cam_pos::WorldPoint{T}, R::RotZYX, world_pt::WorldPoint{T′}) where {T<:Length,T′<:Length}
    # Translate to camera-relative coordinates
    relative_pt = world_pt .- cam_pos

    # Apply inverse rotation (transpose of rotation matrix)
    # This transforms from world frame to camera frame
    cam_vec = R' * relative_pt

    return CameraPoint(cam_vec)
end

"""
    cam_pt_to_world_pt(cam_pos::WorldPoint, R::RotZYX, cam_pt::CameraPoint) -> WorldPoint

Transform a point from camera coordinates to world coordinates.

# Arguments
- `cam_pos::WorldPoint`: Camera position in world coordinates
- `cam_rot::RotZYX`: Camera orientation (ZYX Euler angles: yaw, pitch, roll)
- `cam_pt::CameraPoint`: Point to transform in camera coordinates

# Returns
- `WorldPoint`: Point in world coordinate system

# Algorithm
1. Rotate point by camera rotation to get world-relative coordinates
2. Translate by camera position to get absolute world coordinates

# Examples
```jldoctest
using RunwayLib, Unitful.DefaultSymbols, Rotations
# Transform camera point back to world coordinates
cam_pos = WorldPoint(10.0m, 20m, 30m)
cam_rot = RotZYX(roll=0.0rad, pitch=0rad, yaw=0rad)  # No rotation
cam_pt = CameraPoint(1.0m, 2m, 3m)

world_pt = cam_pt_to_world_pt(cam_pos, cam_rot, cam_pt)

# output

3-element WorldPoint{Float64{m}} with indices SOneTo(3):
 11.0 m
 22.0 m
 33.0 m
```
"""
function cam_pt_to_world_pt(cam_pos::WorldPoint, R::RotZYX, cam_pt::CameraPoint)
    # Convert to SVector for rotation
    cam_vec = SVector(cam_pt)

    # Apply rotation to transform from camera frame to world frame
    world_vec = R * cam_vec

    # Create world point and translate by camera position
    relative_pt = WorldPoint(world_vec)

    return cam_pos .+ relative_pt
end
