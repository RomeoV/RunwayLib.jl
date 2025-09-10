"""
Precompile workloads for RunwayLib

This file contains representative workloads that cover the main use cases of RunwayLib
to improve package loading time through precompilation.
"""

import PrecompileTools
PrecompileTools.@compile_workload begin
    using StaticArrays: SA
    using Rotations: RotZYX
    using Unitful: m, NoUnits, ustrip

    # Define typical runway corners (rectangular runway)
    runway_corners_vec = [
        WorldPoint(0.0m, -50.0m, 0.0m),
        WorldPoint(0.0m, 50.0m, 0.0m),
        WorldPoint(3000.0m, 50.0m, 0.0m),
        WorldPoint(3000.0m, -50.0m, 0.0m),
    ]
    runway_corners_SA = SA[runway_corners_vec...]

    # Camera configuration - only CameraMatrix with :offset
    cam_config = CameraMatrix(CAMERA_CONFIG_OFFSET)

    # Define typical aircraft position and orientation
    aircraft_pos = WorldPoint(-3300.0m, 0.0m, 80.0m)
    aircraft_rot = RotZYX(roll=0.0, pitch=0.0, yaw=0.0)

    # Precompile projection and pose estimation functions
    for runway_corners in [runway_corners_vec, runway_corners_SA]
        # Generate typical projections
        projections = [
            project(aircraft_pos, aircraft_rot, corner, cam_config)
            for corner in runway_corners
        ]

        # Skip 6DOF pose estimation precompilation for now
        pose6dof = estimatepose6dof(runway_corners, projections, cam_config)

        # Skip 3DOF pose estimation precompilation for now
        pose3dof = estimatepose3dof(runway_corners, projections, aircraft_rot, cam_config)
    end
end
