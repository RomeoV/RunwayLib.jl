using Test
using RunwayLib
import RunwayLib: PointFeatures, LineFeatures, NO_LINES, pose_optimization_objective_points, pose_optimization_objective_lines
using Rotations
import Rotations: params
using LinearAlgebra
using Unitful, Unitful.DefaultSymbols
using StaticArrays
using Random

@testset "Pose Estimation" begin
    @testset "Static Array Support" begin
        # Shared setup for static array tests
        runway_corners = SA[
            WorldPoint(0.0m, 25.0m, 0.0m),
            WorldPoint(0.0m, -25.0m, 0.0m),
            WorldPoint(1000.0m, 25.0m, 0.0m),
            WorldPoint(1000.0m, -25.0m, 0.0m)
        ]
        world_lines = SA[(runway_corners[1], runway_corners[3]),
            (runway_corners[2], runway_corners[4])]

        true_pos = WorldPoint(-500.0m, 10.0m, 100.0m)
        true_rot = RotZYX(0.02, 0.1, -0.01)

        observed_corners = map(c -> project(true_pos, true_rot, c), runway_corners)
        observed_lines = map(world_lines) do (p1, p2)
            proj1 = project(true_pos, true_rot, p1)
            proj2 = project(true_pos, true_rot, p2)
            getline(proj1, proj2)
        end

        @testset "Point Features" begin
            point_features = PointFeatures(runway_corners, observed_corners,
                CAMERA_CONFIG_OFFSET, SMatrix{8,8}(1.0I))
            result = pose_optimization_objective_points(true_pos, true_rot, point_features)
            @test result isa SVector
            @test length(result) == 2 * length(runway_corners)
        end

        @testset "Line Features" begin
            line_features = RunwayLib.LineFeatures(world_lines, observed_lines,
                CAMERA_CONFIG_OFFSET, SMatrix{6,6}(1.0I))
            result = RunwayLib.pose_optimization_objective_lines(true_pos, true_rot, line_features)
            @test result isa SVector
            @test length(result) == 3 * length(world_lines)
        end

        @testset "Combined Point and Line Features" begin
            point_features = PointFeatures(runway_corners, observed_corners,
                CAMERA_CONFIG_OFFSET, SMatrix{8,8}(1.0I))
            line_features = RunwayLib.LineFeatures(world_lines, observed_lines,
                CAMERA_CONFIG_OFFSET, SMatrix{6,6}(1.0I))
            ps = RunwayLib.PoseOptimizationParams6DOF(point_features, line_features)
            optvar = [true_pos.x / 1m, true_pos.y / 1m, true_pos.z / 1m,
                true_rot.theta1, true_rot.theta2, true_rot.theta3]

            result = RunwayLib.pose_optimization_objective(optvar, ps)
            @test result isa SVector skip = true  # currently we cast to Array to make sure to get a mutable jacobian
            @test length(result) == 2 * length(runway_corners) + 3 * length(world_lines)
        end
    end

    # Define standard runway corners (4 points forming a rectangle)
    runway_corners = [
        WorldPoint(0.0m, 25.0m, 0.0m),      # near left
        WorldPoint(0.0m, -25.0m, 0.0m),     # near right
        WorldPoint(1000.0m, 25.0m, 0.0m),   # far left
        WorldPoint(1000.0m, -25.0m, 0.0m),  # far right
    ]
    runway_lines = [(runway_corners[1], runway_corners[3]), (runway_corners[2], runway_corners[4])]

    @testset "Pose Estimation - Lines" begin
        true_pos = WorldPoint(-500.0m, 10.0m, 100.0m)
        true_rot = RotZYX(roll=0.02, pitch=0.1, yaw=-0.01)

        observed_lines = map(runway_lines) do (p1, p2)
            getline(project(true_pos, true_rot, p1), project(true_pos, true_rot, p2))
        end

        @testset "Line Objective Function" begin
            line_features = LineFeatures(runway_lines, observed_lines,
                CAMERA_CONFIG_OFFSET, SMatrix{6,6}(1.0I))
            result = pose_optimization_objective_lines(true_pos, true_rot, line_features)
            @test result isa AbstractVector
            @test length(result) == 3 * length(runway_lines)
            @test all(abs.(result) .< 1e-10)  # Should be near zero at true pose
        end

        @testset "Lines Improve Accuracy" begin
            rng = MersenneTwister(123)
            # Add noise to point observations
            noisy_projections = [
                proj + ProjectionPoint(2.0px * randn(rng, 2))
                for proj in [project(true_pos, true_rot, c) for c in runway_corners]
            ]

            guess_pos = [true_pos.x + 100.0m, true_pos.y - 20.0m, true_pos.z + 30.0m]
            guess_rot = [true_rot.theta1 + 0.05, true_rot.theta2 - 0.08, true_rot.theta3 + 0.03]rad

            # Points-only estimation
            point_noise = SMatrix{8,8}(diagm(fill(2.0^2, 8)))
            point_features = PointFeatures(runway_corners, noisy_projections,
                CAMERA_CONFIG_OFFSET, point_noise)
            result_points = estimatepose3dof(point_features, RunwayLib.NO_LINES,
                true_rot; initial_guess_pos=guess_pos)

            # Points + perfect lines estimation
            line_noise = SMatrix{6,6}(diagm(fill(0.5^2, 6)))
            line_features = LineFeatures(
                runway_lines, observed_lines,
                CAMERA_CONFIG_OFFSET, line_noise
            )
            result_combined = estimatepose3dof(
                point_features, line_features, true_rot;
                initial_guess_pos=guess_pos)

            # Combined should be more accurate in crosstrack and height.
            # sidelines dont't give much information about alongtrack.
            err_points = norm(result_points.pos[2:3] - true_pos[2:3])
            err_combined = norm(result_combined.pos[2:3] - true_pos[2:3])
            @test err_combined < err_points
        end
    end

    @testset "Pose Estimation - Offset" begin
        config = CAMERA_CONFIG_OFFSET
        # Ground truth airplane pose
        true_pos = WorldPoint(-500.0m, 10.0m, 100.0m)
        true_rot = RotZYX(roll=0.02, pitch=0.1, yaw=-0.01)

        # Generate perfect projections
        true_projections = [project(true_pos, true_rot, corner, config) for corner in runway_corners]

        # Create noisy initial guesses
        noisy_pos_guess = [true_pos.x + 100.0m, true_pos.y - 20.0m, true_pos.z + 30.0m]
        noisy_rot_guess = [true_rot.theta1 + 0.05, true_rot.theta2 - 0.08, true_rot.theta3 + 0.03]rad

        @testset "6DOF Estimation" begin
            result = estimatepose6dof(
                runway_corners, true_projections, config;
                initial_guess_pos=noisy_pos_guess,
                initial_guess_rot=noisy_rot_guess
            )
            @test norm(result.pos - true_pos) < (eps(eltype(result.pos))^(1 // 3))m
            @test result.rot ≈ true_rot
        end

        @testset "3DOF Estimation" begin
            result = estimatepose3dof(
                runway_corners, true_projections, true_rot, config;
                initial_guess_pos=noisy_pos_guess
            )
            @test norm(result.pos - true_pos) < (eps(eltype(result.pos))^(1 // 3))m
            @test result.rot ≈ true_rot
        end
    end

    @testset "Multiple Test Cases" begin
        # Test multiple ground truth scenarios
        test_cases = [
            (pos=WorldPoint(-300.0m, 0.0m, 80.0m), rot=RotZYX(0.0, 0.05, 0.0)),
            (pos=WorldPoint(-1200.0m, 30.0m, 200.0m), rot=RotZYX(-0.03, 0.12, 0.02)),
            (pos=WorldPoint(-600.0m, -15.0m, 120.0m), rot=RotZYX(0.01, 0.08, -0.01)),
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
                        initial_guess_pos=noisy_pos, initial_guess_rot=noisy_rot
                    )
                    @test norm(result.pos - case.pos) < (eps(eltype(result.pos))^(1 // 3))m
                    @test Rotations.params(result.rot) ≈ Rotations.params(case.rot)
                end

                @testset "3DOF" begin
                    result = estimatepose3dof(
                        runway_corners, true_projections, case.rot;
                        initial_guess_pos=noisy_pos
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
        true_rot = RotZYX(roll=0.01, pitch=0.08, yaw=-0.005)

        for (corners, n_points) in [(runway_corners_8, 8), (runway_corners_16, 16)]
            @testset "$n_points keypoints" begin
                # Generate projections
                camconfig = CAMERA_CONFIG_OFFSET
                true_projections = [project(true_pos, true_rot, corner, camconfig) for corner in corners]

                # Initial guesses
                noisy_pos = [true_pos.x + 150.0m, true_pos.y - 30.0m, true_pos.z + 40.0m]
                noisy_rot = [true_rot.theta1 + 0.08, true_rot.theta2 - 0.12, true_rot.theta3 + 0.06]rad

                @testset "6DOF with $n_points points" begin
                    result = estimatepose6dof(
                        corners, true_projections, camconfig;
                        initial_guess_pos=noisy_pos,
                        initial_guess_rot=noisy_rot
                    )
                    @test norm(result.pos - true_pos) < (eps(eltype(result.pos))^(1 // 3))m
                    @test result.rot ≈ true_rot
                end

                @testset "3DOF with $n_points points" begin
                    result = estimatepose3dof(
                        corners, true_projections, true_rot, camconfig;
                        initial_guess_pos=noisy_pos
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
            #! format: off
            SA[-1100     0 2048
                   0 -1100 1536
                   0     0    1] * 1.0px,
            4096.0px, 3072.0px
            #! format: on
        )

        # Ground truth airplane pose
        true_pos = WorldPoint(-1800.0m, 5.0m, 120.0m)
        true_rot = RotZYX(roll=0.015, pitch=0.08, yaw=-0.005)

        # Generate perfect projections using the custom camera matrix
        true_projections = [project(true_pos, true_rot, corner, custom_camera_matrix)
                            for corner in runway_corners]

        # Create significantly off initial guesses to test convergence
        noisy_pos_guess = [true_pos.x + 200.0m, true_pos.y - 30.0m, true_pos.z + 50.0m]
        noisy_rot_guess = [true_rot.theta1 + 0.1, true_rot.theta2 - 0.12, true_rot.theta3 + 0.08]rad

        @testset "6DOF with Custom Camera Matrix" begin
            result = estimatepose6dof(
                runway_corners, true_projections, custom_camera_matrix;
                initial_guess_pos=noisy_pos_guess,
                initial_guess_rot=noisy_rot_guess
            )
            @test norm(result.pos - true_pos) < 1m
            @test params(result.rot) ≈ params(true_rot) rtol = 0.01
        end

        @testset "3DOF with Custom Camera Matrix" begin
            result = estimatepose3dof(
                runway_corners, true_projections, true_rot, custom_camera_matrix;
                initial_guess_pos=noisy_pos_guess
            )
            @test norm(result.pos - true_pos) < 1m
            @test params(result.rot) ≈ params(true_rot) rtol = 0.01
        end
    end
end
