from poseest import (
    estimate_pose_3dof, estimate_pose_6dof, WorldPoint, ProjectionPoint, Rotation,
    CameraConfig, DiagonalCovariance
)

# Define runway corners in world coordinates (meters)
runway_corners = [
    WorldPoint(0.0, -50.0, 0.0),      # near left
    WorldPoint(0.0, 50.0, 0.0),       # near right
    WorldPoint(1000.0, -50.0, 0.0),   # far left
    WorldPoint(1000.0, 50.0, 0.0)     # far right
]

# Observed projections in image coordinates (pixels)
projections = [
    ProjectionPoint(320.0, 240.0),
    ProjectionPoint(380.0, 240.0),
    ProjectionPoint(380.0, 280.0),
    ProjectionPoint(320.0, 280.0)
]

# Diagonal covariance (pixel variances, not std deviations)
pixel_variances = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
covariance = DiagonalCovariance(variances=pixel_variances)

# 6-DOF estimation (position + orientation)
result_6dof = estimate_pose_6dof(
    runway_corners=runway_corners,
    projections=projections,
    camera_config=CameraConfig.OFFSET,
    covariance=covariance
)

# 3-DOF estimation (position only, known orientation)
known_rotation = Rotation(yaw=-0.01, pitch=0.1, roll=0.02)
result_3dof = estimate_pose_3dof(
    runway_corners=runway_corners,
    projections=projections,
    known_rotation=known_rotation,
    camera_config=CameraConfig.OFFSET,
    covariance=covariance
)

print(f"6-DOF: position={result_6dof.position}, rotation={result_6dof.rotation}")
print(f"3-DOF: position={result_3dof.position}")
