"""
Tests for covariance specification functionality in C API and optimization.

This module tests:
1. Covariance data parsing from C pointers
2. Different covariance types (scalar, diagonal, block diagonal, full matrix)
3. Integration with optimization infrastructure
4. Error handling for invalid covariance specifications
"""

using Test
using RunwayLib
using RunwayLib: px  # Import the pixel unit from RunwayLib
using Distributions: Normal, MvNormal
using LinearAlgebra: Diagonal, isposdef
using Statistics: std, cov  # Import std and cov functions
using LinearAlgebra: I  # Import identity matrix
using StaticArrays: SA
using Unitful: m
using ProbabilisticParameterEstimators: CorrGaussianNoiseModel, covmatrix

@testset "Covariance Specification Tests" begin

    # Test data setup
    runway_corners = [
        WorldPoint(1000.0m, -50.0m, 0.0m),
        WorldPoint(1000.0m, 50.0m, 0.0m),
        WorldPoint(3000.0m, 50.0m, 0.0m),
        WorldPoint(3000.0m, -50.0m, 0.0m),
    ]
    
    # Create some synthetic projections
    projections = [
        ProjectionPoint(100.0px, 200.0px),
        ProjectionPoint(150.0px, 200.0px),
        ProjectionPoint(150.0px, 250.0px),
        ProjectionPoint(100.0px, 250.0px),
    ]

    @testset "Covariance Data Parsing" begin
        
        @testset "Default Covariance" begin
            # Test explicit default covariance type
            noise_model = RunwayLib.parse_covariance_data(
                RunwayLib.COV_DEFAULT, Ptr{Cdouble}(0), 4
            )
            
            @test size(noise_model) == (8, 8)  # 4 points * 2 coords
            # Should match the default noise model
            default_model = RunwayLib._defaultnoisemodel(projections)
            @test all(sqrt.(diag(noise_model)) .≈ std.(default_model.noisedistributions))
        end
        
        @testset "Scalar Covariance" begin
            # Test scalar covariance parsing
            noise_std = 2.5
            cov_data = [noise_std]
            
            noise_model = RunwayLib.parse_covariance_data(
                RunwayLib.COV_SCALAR, pointer(cov_data), 4
            )
            
            @test size(noise_model) == (8, 8)  # 4 points * 2 coords
            @test all(sqrt.(diag(noise_model)) .≈ noise_std)
        end

        @testset "Diagonal Covariance" begin
            # Test diagonal covariance parsing
            variances = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]  # 4 points * 2 coords
            
            noise_model = RunwayLib.parse_covariance_data(
                RunwayLib.COV_DIAGONAL_FULL, pointer(variances), 4
            )
            
            @test size(noise_model) == (8, 8)
            expected_stds = sqrt.(variances)
            @test all(sqrt.(diag(noise_model)) .≈ expected_stds)
        end

        @testset "Block Diagonal Covariance" begin
            # Test block diagonal covariance parsing
            # 4 keypoints, each with 2x2 covariance matrix (stored row-major)
            cov_data = [
                # Point 1: [1.0 0.1; 0.1 1.0]
                1.0, 0.1, 0.1, 1.0,
                # Point 2: [2.0 0.2; 0.2 2.0]
                2.0, 0.2, 0.2, 2.0,
                # Point 3: [1.5 0.0; 0.0 1.5]
                1.5, 0.0, 0.0, 1.5,
                # Point 4: [2.5 -0.3; -0.3 2.5]
                2.5, -0.3, -0.3, 2.5
            ]
            
            noise_model = RunwayLib.parse_covariance_data(
                RunwayLib.COV_BLOCK_DIAGONAL, pointer(cov_data), 4
            )
            
            @test size(noise_model) == (8, 8)

            # Check that covariance matrices are correctly reconstructed
            @test noise_model[1:2, 1:2] ≈ [1.0 0.1; 0.1 1.0]
            @test noise_model[3:4, 3:4] ≈ [2.0 0.2; 0.2 2.0]
            @test noise_model[5:6, 5:6] ≈ [1.5 0.0; 0.0 1.5]
            @test noise_model[7:8, 7:8] ≈ [2.5 -0.3; -0.3 2.5]
        end

        @testset "Full Matrix Covariance" begin
            # Test full matrix covariance parsing
            # 4 points * 2 coords = 8x8 matrix
            matrix_size = 8
            full_cov = Matrix{Float64}(I, matrix_size, matrix_size)
            # Add some off-diagonal elements
            full_cov[1,2] = full_cov[2,1] = 0.1
            full_cov[3,4] = full_cov[4,3] = 0.2
            
            # Flatten to row-major order
            cov_data = vec(full_cov')
            
            noise_model = RunwayLib.parse_covariance_data(
                RunwayLib.COV_FULL_MATRIX, pointer(cov_data), 4
            )
            
            @test size(noise_model) == (8, 8)
            @test noise_model ≈ full_cov
        end
    end

    @testset "Error Handling" begin
        
        @testset "Invalid Scalar Covariance" begin
            # Test negative noise standard deviation
            cov_data = [-1.0]
            @test_throws ArgumentError RunwayLib.parse_covariance_data(
                RunwayLib.COV_SCALAR, pointer(cov_data), 4
            )
        end

        @testset "Invalid Diagonal Covariance" begin
            # Test negative variance
            variances = [1.0, -1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
            @test_throws ArgumentError RunwayLib.parse_covariance_data(
                RunwayLib.COV_DIAGONAL_FULL, pointer(variances), 4
            )
        end

        @testset "Non-positive-definite Block Diagonal" begin
            # Test non-positive-definite 2x2 matrix
            cov_data = [
                1.0, 2.0, 2.0, 1.0,  # This matrix has negative determinant
                2.0, 0.0, 0.0, 2.0,
                1.5, 0.0, 0.0, 1.5,
                2.5, 0.0, 0.0, 2.5
            ]
            @test_throws ArgumentError RunwayLib.parse_covariance_data(
                RunwayLib.COV_BLOCK_DIAGONAL, pointer(cov_data), 4
            )
        end

        @testset "Non-positive-definite Full Matrix" begin
            # Test non-positive-definite full matrix
            matrix_size = 8
            full_cov = -Matrix{Float64}(I, matrix_size, matrix_size)  # Negative definite
            cov_data = vec(full_cov')
            
            @test_throws ArgumentError RunwayLib.parse_covariance_data(
                RunwayLib.COV_FULL_MATRIX, pointer(cov_data), 4
            )
        end
    end

    @testset "Integration with Optimization" begin
        
        @testset "Dense Matrix Conversion" begin
            # Test that optimization parameters use dense matrices
            noise_std = 2.0
            noise_dists = [Normal(0.0, noise_std) for _ in 1:8]
            noise_model = UncorrGaussianNoiseModel(noise_dists)
            
            ps = PoseOptimizationParams6DOF(
                runway_corners, projections, 
                CAMERA_CONFIG_OFFSET, noise_model
            )
            
            # Check that Linv is a dense Matrix
            @test isa(ps.Linv, Matrix)
            @test size(ps.Linv) == (8, 8)
        end

        @testset "Scalar Covariance Integration" begin
            # Test that scalar covariance works with optimization
            noise_std = 1.5
            cov_data = [noise_std]
            
            noise_model = RunwayLib.parse_covariance_data(
                RunwayLib.COV_SCALAR, pointer(cov_data), 4
            )
            
            ps = PoseOptimizationParams6DOF(
                runway_corners, projections,
                CAMERA_CONFIG_OFFSET, noise_model |> Matrix
            )
            
            # Test that we can call the objective function
            test_params = [0.0, 0.0, 100.0, 0.0, 0.0, 0.0]  # Some reasonable guess
            residuals = RunwayLib.pose_optimization_objective(test_params, ps)
            
            @test length(residuals) == 8
            @test all(isfinite.(residuals))
        end

        @testset "Block Diagonal Covariance Integration" begin
            # Test block diagonal covariance with optimization
            cov_data = [
                2.0, 0.1, 0.1, 2.0,  # Point 1
                1.5, 0.0, 0.0, 1.5,  # Point 2
                3.0, -0.2, -0.2, 3.0,  # Point 3
                2.5, 0.15, 0.15, 2.5   # Point 4
            ]
            
            noise_model = RunwayLib.parse_covariance_data(
                RunwayLib.COV_BLOCK_DIAGONAL, pointer(cov_data), 4
            )
            
            ps = PoseOptimizationParams6DOF(
                runway_corners, projections,
                CAMERA_CONFIG_OFFSET, noise_model |> Matrix
            )
            
            # Test objective function call
            test_params = [-1000.0, 0.0, 100.0, 0.01, 0.02, 0.03]
            residuals = RunwayLib.pose_optimization_objective(test_params, ps)
            
            @test length(residuals) == 8
            @test all(isfinite.(residuals))
        end
    end

    @testset "Performance and Memory" begin
        
        @testset "Dense Matrix Performance" begin
            # Test that dense matrices don't cause performance issues
            large_noise_model = UncorrGaussianNoiseModel([Normal(0.0, 1.0) for _ in 1:8])  # 4 points * 2 coords
            
            ps = PoseOptimizationParams6DOF(
                runway_corners, projections,  # Use all 4 corners
                CAMERA_CONFIG_OFFSET, large_noise_model
            )
            
            @test isa(ps.Linv, Matrix)
            # Simple performance test - should complete quickly
            test_params = zeros(6)
            @time begin
                for _ in 1:100
                    RunwayLib.pose_optimization_objective(test_params, ps)
                end
            end
        end
    end
end
