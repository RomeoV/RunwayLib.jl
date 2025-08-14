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

    # Helper function to test pose accuracy
    function test_pose_accuracy(result, true_pos, true_rot; pos_tol=1e-6m, rot_tol=1e-8)
        @test norm(result.pos - true_pos) < pos_tol
        @test result.rot ≈ true_rot atol=rot_tol
    end

    @testset "Pose Estimation - $config_name" for (config, config_name) in [
        (CAMERA_CONFIG_CENTERED, "Centered"),
        (CAMERA_CONFIG_OFFSET, "Offset")
    ]
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
            test_pose_accuracy(result, true_pos, true_rot)
        end

        @testset "3DOF Estimation" begin
            result = estimatepose3dof(
                runway_corners, true_projections, true_rot, config;
                initial_guess_pos = noisy_pos_guess
            )
            @test norm(result.pos - true_pos) < 1e-6m
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
                true_projections = [project(case.pos, case.rot, corner, CAMERA_CONFIG_OFFSET) for corner in runway_corners]
                
                # Large initial errors
                noisy_pos = [case.pos.x + 200.0m, case.pos.y - 50.0m, case.pos.z + 80.0m]
                noisy_rot = [case.rot.theta1 + 0.1, case.rot.theta2 - 0.15, case.rot.theta3 + 0.08]rad
                
                @testset "6DOF" begin
                    result = estimatepose6dof(
                        runway_corners, true_projections, CAMERA_CONFIG_OFFSET;
                        initial_guess_pos = noisy_pos, initial_guess_rot = noisy_rot
                    )
                    test_pose_accuracy(result, case.pos, case.rot)
                end

                @testset "3DOF" begin
                    result = estimatepose3dof(
                        runway_corners, true_projections, case.rot, CAMERA_CONFIG_OFFSET;
                        initial_guess_pos = noisy_pos
                    )
                    @test norm(result.pos - case.pos) < 1e-6m
                    @test result.rot ≈ case.rot
                end
            end
        end
    end
end