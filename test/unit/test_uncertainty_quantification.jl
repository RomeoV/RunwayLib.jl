using Test
using RunwayPoseEstimation
using Distributions
using LinearAlgebra
using ProbabilisticParameterEstimators
using Random
using Statistics

@testset "Uncertainty Quantification" begin
    @testset "Noise Model Construction" begin
        # Test uncorrelated Gaussian noise model
        pixel_stds = [2.0, 2.0, 3.0, 3.0, 2.5, 2.5, 2.8, 2.8]  # 4 corners × 2 coordinates
        noise_dists = [Normal(0.0, std) for std in pixel_stds]
        noise_model = UncorrGaussianNoiseModel(noise_dists)
        
        @test length(noise_model.noisedistributions) == 8
        @test all(isa.(noise_model.noisedistributions, Normal))
        
        # Test covariance matrix construction
        cov_matrix = covmatrix(noise_model)
        @test size(cov_matrix) == (8, 8)
        @test isa(cov_matrix, Diagonal)
        @test all(diag(cov_matrix) .> 0)  # Positive variances
        
        # Test multivariate noise distribution
        mv_noise = mvnoisedistribution(noise_model)
        @test isa(mv_noise, MvNormal)
        @test length(mv_noise) == 8
    end
    
    @testset "Correlated Noise Model" begin
        # Test correlated noise between corner observations
        n_obs = 8  # 4 corners × 2 coordinates
        correlation_matrix = Matrix{Float64}(I, n_obs, n_obs)
        
        # Add some correlation between x and y coordinates of same corner
        for i in 1:2:n_obs-1
            correlation_matrix[i, i+1] = 0.3
            correlation_matrix[i+1, i] = 0.3
        end
        
        # Scale to get covariance matrix
        stds = fill(2.0, n_obs)
        cov_matrix = Diagonal(stds) * correlation_matrix * Diagonal(stds)
        
        corr_noise_model = CorrGaussianNoiseModel(MvNormal(zeros(n_obs), cov_matrix))
        
        @test isa(corr_noise_model.noisedistribution, MvNormal)
        @test length(corr_noise_model.noisedistribution) == n_obs
        
        # Test covariance matrix extraction
        extracted_cov = covmatrix(corr_noise_model)
        @test size(extracted_cov) == (n_obs, n_obs)
        @test extracted_cov ≈ cov_matrix
    end
    
    @testset "Uncertainty Propagation" begin
        # Test linear approximation for uncertainty propagation
        # Simple test case: f(x) = 2*x, so uncertainty should scale by factor of 2
        
        # Mock parameter distribution
        param_mean = [1.0, 2.0, 3.0]
        param_cov = Diagonal([0.1, 0.2, 0.3])
        param_dist = MvNormal(param_mean, param_cov)
        
        # Simple linear function for testing
        test_function(x, θ) = 2.0 * θ[1] + θ[2] + 0.5 * θ[3]
        
        # Test that function evaluation works
        xs = [1.0]  # Dummy input
        result = test_function(xs[1], param_mean)
        expected = 2.0 * 1.0 + 2.0 + 0.5 * 3.0
        @test result ≈ expected
        
        # Test uncertainty propagation concept
        # For linear function f(θ) = a'θ, Var[f(θ)] = a' Σ a
        a = [2.0, 1.0, 0.5]  # Coefficients from test_function
        expected_variance = a' * param_cov * a
        @test expected_variance > 0
    end
    
    @testset "Pixel Uncertainty to Pose Uncertainty" begin
        # Test the concept of propagating pixel uncertainties to pose uncertainties
        
        # Mock pixel uncertainties (4 corners × 2 coordinates)
        pixel_vars = [4.0, 4.0, 9.0, 9.0, 6.25, 6.25, 7.84, 7.84]  # σ² values
        pixel_noise_model = UncorrGaussianNoiseModel([Normal(0.0, sqrt(var)) for var in pixel_vars])
        
        # Mock pose parameters (6-DOF: 3 position + 3 rotation)
        pose_prior = MvNormal(zeros(6), Diagonal([1000.0, 100.0, 100.0, 0.1, 0.1, 0.1]))
        
        @test length(pose_prior) == 6
        @test all(diag(cov(pose_prior)) .> 0)
        
        # Test that we can construct the estimation problem
        # (Actual estimation would use ProbabilisticParameterEstimators)
        estimator = LinearApproxEstimator()
        @test isa(estimator, LinearApproxEstimator)
        
        # Mock runway corners and observations for testing
        runway_corners = [
            WorldPoint(0.0, 25.0, 0.0),
            WorldPoint(0.0, -25.0, 0.0),
            WorldPoint(1000.0, 25.0, 0.0),
            WorldPoint(1000.0, -25.0, 0.0)
        ]
        
        observed_corners = [
            ProjectionPoint(-100.0, 50.0),
            ProjectionPoint(-100.0, -50.0),
            ProjectionPoint(100.0, 40.0),
            ProjectionPoint(100.0, -40.0)
        ]
        
        @test length(runway_corners) == length(observed_corners)
        @test all(isa.(runway_corners, WorldPoint))
        @test all(isa.(observed_corners, ProjectionPoint))
    end
    
    @testset "Calibration Assessment" begin
        # Test calibration checking for uncertainty estimates

        # Mock predicted distributions and ground truth values
        rng = MersenneTwister(501)
        n_samples = 100
        true_values = randn(rng, n_samples)

        # Well-calibrated predictions (correct uncertainty)
        predicted_dists_good = [Normal(val, 1.0) for val in true_values .+ 0.1*randn(rng, n_samples)]

        # Under-confident predictions (too much uncertainty)
        predicted_dists_overconf = [Normal(val, 0.5) for val in true_values .+ 0.1*randn(rng, n_samples)]

        # Over-confident predictions (too little uncertainty)
        predicted_dists_underconf = [Normal(val, 2.0) for val in true_values .+ 0.1*randn(rng, n_samples)]
        
        @test length(predicted_dists_good) == n_samples
        @test all(isa.(predicted_dists_good, Normal))
        
        # Test coverage computation concept
        # For well-calibrated predictions, 68% of true values should fall within 1σ
        coverage_68 = mean([abs(true_val - mean(pred_dist)) <= std(pred_dist) 
                           for (true_val, pred_dist) in zip(true_values, predicted_dists_good)])
        
        # Should be approximately 0.68 for well-calibrated predictions
        @test 0.6 <= coverage_68 <= 0.8  # Allow some Monte Carlo variation
    end
    
    @testset "Noise Model Integration" begin
        # Test integration with ProbabilisticParameterEstimators
        
        # Create simple noise model
        corner_uncertainties = [
            [2.0 0.1; 0.1 2.0],  # bottom_left covariance
            [3.0 0.2; 0.2 3.0],  # bottom_right covariance  
            [2.5 0.0; 0.0 2.5],  # top_left covariance
            [2.8 -0.1; -0.1 2.8] # top_right covariance
        ]
        
        # Create multivariate normal distributions for each corner
        corner_dists = [MvNormal(zeros(2), cov) for cov in corner_uncertainties]
        noise_model = UncorrGaussianNoiseModel(corner_dists)
        
        @test length(noise_model.noisedistributions) == 4
        @test all(isa.(noise_model.noisedistributions, MvNormal))
        
        # Test covariance matrix construction
        full_cov = covmatrix(noise_model)
        @test size(full_cov) == (8, 8)  # 4 corners × 2 coordinates
        
        # Test that block diagonal structure is preserved
        @test isa(full_cov, BlockDiagonal)
    end
end
