using Test
using Distributions
using LinearAlgebra
using RunwayLib
using ProbabilisticParameterEstimators
using JET
using Rotations
using Unitful, Unitful.DefaultSymbols
using StaticArrays

# Import pixel unit properly
# import RunwayLib: pixel
# const px = pixel

@testset "JET Type Stability Tests" begin
    @testset "Projection Functions" begin
        # Test data
        cam_pos = WorldPoint(-500.0m, 10.0m, 100.0m)
        cam_rot = RotZYX(roll = 0.02, pitch = 0.1, yaw = -0.01)
        world_pt = WorldPoint(0.0m, 25.0m, 0.0m)
        
        @testset "project - basic functionality" begin
            # Test with default camera config
            @test_opt project(cam_pos, cam_rot, world_pt)
            
            # Test with explicit camera configs
            @test_opt project(cam_pos, cam_rot, world_pt, CAMERA_CONFIG_CENTERED)
            @test_opt project(cam_pos, cam_rot, world_pt, CAMERA_CONFIG_OFFSET)
        end
    end

    @testset "Pose Estimation Functions" begin
        # Test data for pose estimation
        runway_corners = [
            WorldPoint(0.0m, -50.0m, 0.0m),
            WorldPoint(0.0m, 50.0m, 0.0m),
            WorldPoint(3000.0m, 50.0m, 0.0m),
            WorldPoint(3000.0m, -50.0m, 0.0m)
        ]
        true_pos = WorldPoint(-1300.0m, 0.0m, 80.0m)
        true_rot = RotZYX(roll=0.03, pitch=0.04, yaw=0.05)
        camconfig = CAMERA_CONFIG_OFFSET
        projections = [project(true_pos, true_rot, corner, camconfig) for corner in runway_corners]
        

        # Test the pose optimization objective function
        @testset "pose_optimization_objective" begin
            # Test 6-DOF optimization parameters
            opt_params_6dof = [
                -1500.0, 10.0, 100.0,    # position
                0.02, 0.1, -0.01        # rotation (roll, pitch, yaw)
            ]
            
            # Create noise model for testing
            noise_model = UncorrGaussianNoiseModel(
                reduce(vcat, [SA[Normal(0.0, 2.0), Normal(0.0, 2.0)] for _ in projections])
            )
            
            ps_6dof = PoseOptimizationParams6DOF(
                runway_corners, projections,
                camconfig, cholesky(covmatrix(noise_model)).U
            )
            
            @test_opt RunwayLib.pose_optimization_objective(opt_params_6dof, ps_6dof)
            
            # Test 3-DOF optimization parameters
            opt_params_3dof = SA[-500.0, 10.0, 100.0]  # position only
            
            ps_3dof = PoseOptimizationParams3DOF(
                runway_corners, projections,
                camconfig, cholesky(covmatrix(noise_model)).U, true_rot
            )
            
            @test_opt RunwayLib.pose_optimization_objective(opt_params_3dof, ps_3dof)
        end
        
        # Test the pose estimation functions
        @testset "estimatepose functions" begin
            @test_opt estimatepose6dof(
                runway_corners, projections, camconfig
            )
            
            @test_opt estimatepose3dof(
                runway_corners, projections, true_rot, camconfig
            )
        end
    end
end
