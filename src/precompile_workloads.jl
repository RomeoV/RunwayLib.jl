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
    runway_corners = [
        WorldPoint(0.0m, -50.0m, 0.0m),
        WorldPoint(0.0m, 50.0m, 0.0m),
        WorldPoint(3000.0m, 50.0m, 0.0m),
        WorldPoint(3000.0m, -50.0m, 0.0m),
    ]

    # Define typical aircraft position and orientation
    aircraft_pos = WorldPoint(-1300.0m, 0.0m, 80.0m)
    aircraft_rot = RotZYX(roll=0.03, pitch=0.04, yaw=0.05)

    # Camera configurations
    cam_config_centered = CAMERA_CONFIG_CENTERED
    cam_config_offset = CAMERA_CONFIG_OFFSET

    # Precompile projection functions with both camera configs
    for cam_config in [cam_config_centered, cam_config_offset]
        # Generate typical projections with noise
        projections = [
            project(aircraft_pos, aircraft_rot, corner, cam_config) + ProjectionPoint(0.1px, 0.1px)
            for corner in runway_corners
        ]

        # Precompile 6DOF pose estimation
        pose6dof = estimatepose6dof(runway_corners, projections, cam_config)

        # Precompile 3DOF pose estimation
        pose3dof = estimatepose3dof(runway_corners, projections, aircraft_rot, cam_config)
    end
end
