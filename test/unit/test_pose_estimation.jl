using Test
using RunwayLib
using Rotations
using LinearAlgebra
using Unitful, Unitful.DefaultSymbols
using StaticArrays

@testset "Pose Estimation" begin
    # Define standard runway corners (4 points forming a rectangle)
    runway_corners = [
        WorldPoint(0.0m, 25.0m, 0.0m),      # near left  
        WorldPoint(0.0m, -25.0m, 0.0m),     # near right
        WorldPoint(1000.0m, 25.0m, 0.0m),   # far left
        WorldPoint(1000.0m, -25.0m, 0.0m),  # far right
    ]

    @testset "Pose Estimation - Offset" begin
        config = CameraMatrix(CAMERA_CONFIG_OFFSET)
        # Ground truth airplane pose
        true_pos = WorldPoint(-500.0m, 10.0m, 100.0m)
        true_rot = RotZYX(roll = 0.02, pitch = 0.1, yaw = -0.01)

        # Generate perfect projections
        true_projections = [project(true_pos, true_rot, corner, config) for corner in runway_corners]

        # Create noisy initial guesses
        noisy_pos_guess = [true_pos.x + 100.0m, true_pos.y - 20.0m, true_pos.z + 30.0m]
        noisy_rot_guess = [true_rot.theta1 + 0.05, true_rot.theta2 - 0.08, true_rot.theta3 + 0.03]rad

        @testset "6DOF Estimation" begin
            result = estimatepose6dof(
                runway_corners, true_projections, config;
                initial_guess_pos = noisy_pos_guess,
                initial_guess_rot = noisy_rot_guess
            )
            @test norm(result.pos - true_pos) < (eps(eltype(result.pos))^(1 // 3))m
            @test result.rot ≈ true_rot
        end

        @testset "3DOF Estimation" begin
            result = estimatepose3dof(
                runway_corners, true_projections, true_rot, config;
                initial_guess_pos = noisy_pos_guess
            )
            @test norm(result.pos - true_pos) < (eps(eltype(result.pos))^(1 // 3))m
            @test result.rot ≈ true_rot
        end
    end

    @testset "Multiple Test Cases" begin
        # Test multiple ground truth scenarios
        test_cases = [
            (pos = WorldPoint(-300.0m, 0.0m, 80.0m), rot = RotZYX(0.0, 0.05, 0.0)),
            (pos = WorldPoint(-1200.0m, 30.0m, 200.0m), rot = RotZYX(-0.03, 0.12, 0.02)),
            (pos = WorldPoint(-600.0m, -15.0m, 120.0m), rot = RotZYX(0.01, 0.08, -0.01)),
        ]

        for (i, case) in enumerate(test_cases)
            @testset "Case $i" begin
                true_projections = [project(case.pos, case.rot, corner) for corner in runway_corners]
                
                # Large initial errors
                noisy_pos = [case.pos.x + 200.0m, case.pos.y - 50.0m, case.pos.z + 80.0m]
                noisy_rot = [case.rot.theta1 + 0.1, case.rot.theta2 - 0.15, case.rot.theta3 + 0.08]rad
                
                @testset "6DOF" begin
                    result = estimatepose6dof(
                        runway_corners, true_projections;
                        initial_guess_pos = noisy_pos, initial_guess_rot = noisy_rot
                    )
                    @test norm(result.pos - case.pos) < (eps(eltype(result.pos))^(1 // 3))m
                    @test Rotations.params(result.rot) ≈ Rotations.params(case.rot)
                end

                @testset "3DOF" begin
                    result = estimatepose3dof(
                        runway_corners, true_projections, case.rot;
                        initial_guess_pos = noisy_pos
                    )
                    @test norm(result.pos - case.pos) < (eps(eltype(result.pos))^(1 // 3))m
                    @test Rotations.params(result.rot) ≈ Rotations.params(case.rot)
                end
            end
        end
    end

    @testset "Variable Keypoint Numbers" begin
        # Test with 8 keypoints (2x4 grid)
        runway_corners_8 = [
            WorldPoint(0.0m, 25.0m, 0.0m),      # near left  
            WorldPoint(0.0m, -25.0m, 0.0m),     # near right
            WorldPoint(1000.0m, 25.0m, 0.0m),   # far left
            WorldPoint(1000.0m, -25.0m, 0.0m),  # far right
            # Additional points
            WorldPoint(500.0m, 25.0m, 0.0m),    # middle left
            WorldPoint(500.0m, -25.0m, 0.0m),   # middle right
            WorldPoint(750.0m, 0.0m, 0.0m),     # center point
            WorldPoint(250.0m, 0.0m, 0.0m),     # quarter point
        ]

        # Test with 16 keypoints (4x4 grid)
        runway_corners_16 = [
            # First row
            WorldPoint(0.0m, 37.5m, 0.0m), WorldPoint(0.0m, 12.5m, 0.0m),
            WorldPoint(0.0m, -12.5m, 0.0m), WorldPoint(0.0m, -37.5m, 0.0m),
            # Second row
            WorldPoint(333.0m, 37.5m, 0.0m), WorldPoint(333.0m, 12.5m, 0.0m),
            WorldPoint(333.0m, -12.5m, 0.0m), WorldPoint(333.0m, -37.5m, 0.0m),
            # Third row  
            WorldPoint(667.0m, 37.5m, 0.0m), WorldPoint(667.0m, 12.5m, 0.0m),
            WorldPoint(667.0m, -12.5m, 0.0m), WorldPoint(667.0m, -37.5m, 0.0m),
            # Fourth row
            WorldPoint(1000.0m, 37.5m, 0.0m), WorldPoint(1000.0m, 12.5m, 0.0m),
            WorldPoint(1000.0m, -12.5m, 0.0m), WorldPoint(1000.0m, -37.5m, 0.0m),
        ]

        # Ground truth pose
        true_pos = WorldPoint(-400.0m, 5.0m, 90.0m)
        true_rot = RotZYX(roll = 0.01, pitch = 0.08, yaw = -0.005)

        for (corners, n_points) in [(runway_corners_8, 8), (runway_corners_16, 16)]
            @testset "$n_points keypoints" begin
                # Generate projections
                camera_matrix = CameraMatrix(CAMERA_CONFIG_OFFSET)
                true_projections = [project(true_pos, true_rot, corner, camera_matrix) for corner in corners]
                
                # Initial guesses
                noisy_pos = [true_pos.x + 150.0m, true_pos.y - 30.0m, true_pos.z + 40.0m]
                noisy_rot = [true_rot.theta1 + 0.08, true_rot.theta2 - 0.12, true_rot.theta3 + 0.06]rad

                @testset "6DOF with $n_points points" begin
                    result = estimatepose6dof(
                        corners, true_projections, camera_matrix;
                        initial_guess_pos = noisy_pos,
                        initial_guess_rot = noisy_rot
                    )
                    @test norm(result.pos - true_pos) < (eps(eltype(result.pos))^(1 // 3))m
                    @test result.rot ≈ true_rot
                end

                @testset "3DOF with $n_points points" begin
                    result = estimatepose3dof(
                        corners, true_projections, true_rot, camera_matrix;
                        initial_guess_pos = noisy_pos
                    )
                    @test norm(result.pos - true_pos) < (eps(eltype(result.pos))^(1 // 3))m
                    @test result.rot ≈ true_rot
                end
            end
        end
    end

    @testset "Non-Default Camera Matrix" begin
        # Create a custom camera matrix different from CAMERA_CONFIG_OFFSET
        custom_camera_matrix = CameraMatrix{:offset}(
            SA[1100.0 0.0 2048.0; 0.0 1100.0 1536.0; 0.0 0.0 1.0] * px,
            4096.0px, 3072.0px
        )
        
        # Ground truth airplane pose
        true_pos = WorldPoint(-800.0m, 5.0m, 120.0m)
        true_rot = RotZYX(roll = 0.015, pitch = 0.08, yaw = -0.005)

        # Generate perfect projections using the custom camera matrix
        true_projections = [project(true_pos, true_rot, corner, custom_camera_matrix) for corner in runway_corners]

        # Create significantly off initial guesses to test convergence
        noisy_pos_guess = [true_pos.x + 200.0m, true_pos.y - 30.0m, true_pos.z + 50.0m]
        noisy_rot_guess = [true_rot.theta1 + 0.1, true_rot.theta2 - 0.12, true_rot.theta3 + 0.08]rad

        @testset "6DOF with Custom Camera Matrix" begin
            result = estimatepose6dof(
                runway_corners, true_projections, custom_camera_matrix;
                initial_guess_pos = noisy_pos_guess,
                initial_guess_rot = noisy_rot_guess
            )
            @test norm(result.pos - true_pos) < (eps(eltype(result.pos))^(1 // 3))m
            @test result.rot ≈ true_rot
        end

        @testset "3DOF with Custom Camera Matrix" begin
            result = estimatepose3dof(
                runway_corners, true_projections, true_rot, custom_camera_matrix;
                initial_guess_pos = noisy_pos_guess
            )
            @test norm(result.pos - true_pos) < (eps(eltype(result.pos))^(1 // 3))m
            @test result.rot ≈ true_rot
        end
    end
end
