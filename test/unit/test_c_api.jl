using Test
using RunwayLib
using RunwayLib: _ustrip
using StaticArrays
using Rotations
using Unitful
using Unitful.DefaultSymbols
using JET

@testset "C API" begin
    # Test data
    runway_corners = SA[
        WorldPoint(0.0m, -50.0m, 0.0m),
        WorldPoint(0.0m, 50.0m, 0.0m),
        WorldPoint(3000.0m, 50.0m, 0.0m),
        WorldPoint(3000.0m, -50.0m, 0.0m),
    ]

    true_pos = WorldPoint(-2300.0m, 0.0m, 80.0m)
    true_rot = RotZYX(roll=0.01, pitch=0.01, yaw=-0.01)
    camconfig = CAMERA_CONFIG_OFFSET

    projections = [
        project(true_pos, true_rot, corner, camconfig) + ProjectionPoint(5*randn(2)) * px
        for corner in runway_corners
    ]

    @testset "Library Initialization" begin
        # Skip initialization tests as they have side effects
        # and the function exists from previous tests
    end

    @testset "6DOF Pose Estimation @ccallable" begin
        # Create C structs for testing
        runway_corners_ = [corner .|> _ustrip(m) for corner in runway_corners]
        projections_ = [proj .|> _ustrip(px) for proj in projections]

        # Allocate result struct
        result = Ref{RunwayLib.PoseEstimate_C}()

        # Test function call (may return NO_CONVERGENCE due to optimizer issue)
        error_code = RunwayLib.estimate_pose_6dof(
            pointer(runway_corners_), pointer(projections_),
            Cint(length(runway_corners_)), RunwayLib.CAMERA_CONFIG_OFFSET_C, Base.unsafe_convert(Ptr{RunwayLib.PoseEstimate_C}, result)
        )

        # Function should not crash - accept either success or convergence error
        @test error_code == RunwayLib.POSEEST_SUCCESS
        @test result[].position * m ≈ true_pos rtol = 1e-2

        @test_opt RunwayLib.estimate_pose_6dof(
            pointer(runway_corners_), pointer(projections_),
            Cint(length(runway_corners_)), RunwayLib.CAMERA_CONFIG_OFFSET_C, Base.unsafe_convert(Ptr{RunwayLib.PoseEstimate_C}, result)
        )
    end

    @testset "3DOF Pose Estimation @ccallable" begin
        # Create C structs for testing
        runway_corners_ = [corner .|> _ustrip(m) for corner in runway_corners]
        projections_ = [proj .|> _ustrip(px) for proj in projections]

        # Allocate result struct
        result = Ref{RunwayLib.PoseEstimate_C}()

        # Create known rotation for 3DOF
        known_rot_c = Rotations.params(true_rot)

        # Test function call (may return NO_CONVERGENCE due to optimizer issue)
        error_code = RunwayLib.estimate_pose_3dof(
            pointer(runway_corners_), pointer(projections_),
            Cint(length(runway_corners_)), Base.unsafe_convert(Ptr{RunwayLib.RotYPRF64}, Ref(known_rot_c)),
            RunwayLib.CAMERA_CONFIG_OFFSET_C, Base.unsafe_convert(Ptr{RunwayLib.PoseEstimate_C}, result)
        )

        # Function should not crash - accept either success or convergence error
        @test error_code == RunwayLib.POSEEST_SUCCESS
        @test result[].position * m ≈ true_pos rtol = 1e-2

        @test_opt RunwayLib.estimate_pose_3dof(
            pointer(runway_corners_), pointer(projections_),
            Cint(length(runway_corners_)), Base.unsafe_convert(Ptr{RunwayLib.RotYPRF64}, Ref(known_rot_c)),
            RunwayLib.CAMERA_CONFIG_OFFSET_C, Base.unsafe_convert(Ptr{RunwayLib.PoseEstimate_C}, result)
        )
    end

    @testset "Point Projection @ccallable" begin
        # Create input structs
        world_point = runway_corners[1] .|> _ustrip(m)
        position = true_pos .|> _ustrip(m)
        rotation = Rotations.params(true_rot)

        # Allocate result
        result = Ref{RunwayLib.ProjectionPointF64}()

        # Test projection
        error_code = RunwayLib.project_point(
            Base.unsafe_convert(Ptr{RunwayLib.WorldPointF64}, Ref(position)),
            Base.unsafe_convert(Ptr{RunwayLib.RotYPRF64}, Ref(rotation)),
            Base.unsafe_convert(Ptr{RunwayLib.WorldPointF64}, Ref(world_point)),
            RunwayLib.CAMERA_CONFIG_OFFSET_C, Base.unsafe_convert(Ptr{RunwayLib.ProjectionPointF64}, result)
        )

        @test error_code == RunwayLib.POSEEST_SUCCESS

        # Compare with direct Julia projection
        expected = project(true_pos, true_rot, runway_corners[1], camconfig)
        @test abs(result[].x - ustrip(px, expected.x)) < 1.0
        @test abs(result[].y - ustrip(px, expected.y)) < 1.0
    end

    @testset "Error Handling @ccallable" begin
        # Test with insufficient points
        world_points_ = [RunwayLib.WorldPointF64(0.0, 0.0, 0.0)]
        projection_points_ = [RunwayLib.ProjectionPointF64(0.0, 0.0)]
        result = Ref{RunwayLib.PoseEstimate_C}()

        error_code = RunwayLib.estimate_pose_6dof(
            pointer(world_points_), pointer(projection_points_),
            Cint(1), RunwayLib.CAMERA_CONFIG_OFFSET_C, Base.unsafe_convert(Ptr{RunwayLib.PoseEstimate_C}, result)  # Only 1 point, need at least 4
        )
        @test error_code == RunwayLib.POSEEST_ERROR_INVALID_INPUT

        # Test error message function
        msg_ptr = RunwayLib.get_error_message(error_code)
        @test msg_ptr != C_NULL

        # Convert to string and check it's not empty
        msg = unsafe_string(msg_ptr)
        @test !isempty(msg)
    end

    @testset "PoseEstimate_C Struct Conversion" begin
        # Test conversion of complete pose estimate
        julia_pos = WorldPoint(100.0m, -50.0m, 25.0m)
        julia_rot = RotZYX(0.1, 0.2, 0.3)

        # Create C struct
        c_struct = RunwayLib.PoseEstimate_C(
            julia_pos .|> _ustrip(m),
            Rotations.params(julia_rot),
            1.5,  # residual_norm
            Cint(1)   # converged
        )

        # Test conversion back to Julia types
        converted_pos = c_struct.position .* m
        converted_rot = RotZYX(c_struct.rotation...)

        @test converted_pos ≈ julia_pos
        @test converted_rot.theta1 ≈ julia_rot.theta1
        @test converted_rot.theta2 ≈ julia_rot.theta2
        @test converted_rot.theta3 ≈ julia_rot.theta3
        @test c_struct.converged == Cint(1)
        @test c_struct.residual_norm == 1.5
    end
end
