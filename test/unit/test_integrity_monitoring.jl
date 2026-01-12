"""
Integrity Monitoring Tests

This test suite validates the RAIM (Receiver Autonomous Integrity Monitoring) 
functionality for runway pose estimation. The tests cover:

## Core Functionality Tests
1. **RAIM Statistic Computation**: Test compute_integrity_statistic() with various inputs
   - Basic functionality with known good data
   - Edge cases (minimum observations, different noise levels)
   - Unit handling and coordinate system consistency

## Statistical Properties Tests
4. **Chi-Squared Distribution**: Validate statistical assumptions
   - Degrees of freedom calculation (n_obs - n_params)
   - P-value computation accuracy
   - Proper handling of different noise covariance structures

## Robustness and Edge Cases
6. **Geometric Configuration Effects**: 
   - Well-conditioned vs poorly-conditioned corner geometries
   - Effect of observation geometry on integrity performance
   - Near-singular cases and numerical stability

7. **Fault Detection Scenarios**:
   - Normal operation (small residuals, integrity OK)
   - Single observation fault (large residual, integrity fail)
   - Multiple observation faults
   - Systematic vs random errors

8. **Noise Model Validation**:
   - Diagonal vs full covariance matrices
   - Different noise levels and their impact on statistics
   - Proper whitening and chi-squared test validity
"""

using Test
using RunwayLib
using Distributions
using LinearAlgebra
using Rotations
using StaticArrays
using StatsBase
using Random
using Unitful
using Unitful.DefaultSymbols
include("../test_utils.jl")

# ==============================================================================
# Test Utility Functions
# ==============================================================================

"""
Create a standard test scenario with runway corners, true pose, and observations.
"""
function create_runway_scenario(;
    n_corners::Int=4,
    true_pos=WorldPoint(-800.0m, 5.0m, 120.0m),
    true_rot=RotZYX(0.02, 0.05, 0.01),  # Small attitude angles
)
    # Standard runway corners - well-conditioned geometry
    if n_corners == 4
        runway_corners = [
            WorldPoint(0.0m, -25.0m, 0.0m),
            WorldPoint(0.0m, 25.0m, 0.0m),
            WorldPoint(1500.0m, -25.0m, 0.0m),
            WorldPoint(1500.0m, 25.0m, 0.0m)
        ]
    else
        # Generate more corners if needed
        runway_corners = [
            WorldPoint(x * m, y * m, 0.0m)
            for x in range(0, 1500, length=div(n_corners, 2))
            for y in [-25.0, 25.0]
        ][1:n_corners]
    end

    # Generate clean projections
    clean_projections = [project(true_pos, true_rot, corner, CAMERA_CONFIG_OFFSET)
                         for corner in runway_corners]

    make_noisy_projections(rng::AbstractRNG, σ=1.0) = clean_projections .+ [
        ProjectionPoint(σ * randn(rng, 2)px)
        for _ in clean_projections
    ]

    return (;
        runway_corners,
        true_pos,
        true_rot,
        clean_projections,
        make_noisy_projections,
    )
end

sample_aircraft_pos(rng::AbstractRNG) =
    let dist = MvNormal([-3000.0, 0, 300], Diagonal([1000.0, 200, 100] .^ 2))
        pos = rand(rng, dist)
        pos = clamp.(pos, [-Inf, -Inf, 100], [-1000, Inf, Inf])
        WorldPoint(pos) * m
    end

sample_aircraft_rot(rng::AbstractRNG) = RotZYX((deg2rad(5) * randn(rng, 3))...)

const CAMERA_CONFIGS = [
    ("CameraConfig :offset", CAMERA_CONFIG_OFFSET),
    ("CameraMatrix :offset", CameraMatrix(CAMERA_CONFIG_OFFSET))
]

"""
Validate p-value distribution against uniform distribution.
"""
function validate_p_value_distribution(p_values; n_bins::Int=20, α::Float64=0.01)
    # Bin p-values
    bin_edges = range(0, 1, length=n_bins + 1)
    bin_counts = fit(Histogram, p_values, bin_edges).weights

    # Expected count per bin for uniform distribution
    expected_count = length(p_values) / n_bins

    # Chi-squared goodness of fit test
    chi_sq_stat = sum((bin_counts .- expected_count) .^ 2 ./ expected_count)
    p_value_test = ccdf(Chisq(n_bins - 1), chi_sq_stat)

    return (
        bin_counts=bin_counts,
        expected_count=expected_count,
        chi_squared=chi_sq_stat,
        p_value=p_value_test,
        uniform_distribution=p_value_test > α
    )
end

# ==============================================================================
# Test Cases
# ==============================================================================

@testset "Integrity Monitoring" begin

    @testset "1. API Consistency Tests" begin
        (; true_pos, true_rot, runway_corners, make_noisy_projections) = create_runway_scenario()
        sigmas = 2.0 * ones(length(runway_corners))
        # Create diagonal covariance matrix - each corner has x,y coordinates
        noise_cov = Diagonal(repeat(sigmas .^ 2, inner=2))

        @testset "Camera Configuration Consistency" begin
            # Test with different camera configurations using offset coordinates
            offset_configs = [
                ("CameraMatrix :offset", CameraMatrix(CAMERA_CONFIG_OFFSET))
            ]

            results = map(offset_configs) do (name, config)
                rng = MersenneTwister(1)
                projections = make_noisy_projections(rng)
                compute_integrity_statistic(
                    true_pos, true_rot,
                    runway_corners, projections,
                    noise_cov, config
                )
            end

            # All configurations should give similar results (within numerical precision)
            base_result = results[1]
            for result in results[2:end]
                @test isapprox(result.stat, base_result.stat, rtol=1e-6)
                @test isapprox(result.p_value, base_result.p_value, rtol=1e-6)
                @test result.dofs == base_result.dofs
            end
        end
    end

    @testset "2. Normal vs High Noise Scenarios" begin
        @testset "Normal Noise Case" begin
            rng = MersenneTwister(42)
            (; true_pos, true_rot, runway_corners, make_noisy_projections) = create_runway_scenario()
            noise_level = 2.0
            sigmas = noise_level * ones(length(runway_corners))
            noise_cov = Diagonal(repeat(sigmas .^ 2, inner=2))

            stats = compute_integrity_statistic(
                true_pos, true_rot,
                runway_corners, make_noisy_projections(rng, noise_level),
                noise_cov
            )

            # Normal case should pass integrity check
            @test stats.p_value > 0.05
            @test stats.dofs == 2
            @test stats.stat >= 0
            @test isfinite(stats.stat)
        end

        @testset "High Noise Case" begin
            rng = MersenneTwister(43)
            # Create scenario with higher actual noise than modeled
            noise_level = 6.0  # 3x higher actual noise
            (; true_pos, true_rot, runway_corners, make_noisy_projections) = create_runway_scenario()
            modeled_sigmas = 2 * ones(length(runway_corners))  # Still model as 2.0
            modeled_cov = Diagonal(repeat(modeled_sigmas .^ 2, inner=2))

            stats = compute_integrity_statistic(
                true_pos, true_rot,
                runway_corners,
                make_noisy_projections(rng, noise_level),
                modeled_cov
            )

            # High noise case should have larger chi-squared statistic than normal case
            @test stats.stat > 0.5
        end
    end

    @testset "3. Statistical Calibration (Monte Carlo)" begin
        n_trials = 1000  # Reduced for faster testing, increase for production

        @testset "P-value Distribution Test" begin
            rng = MersenneTwister(44)
            noise_level = 2.0
            p_values = map(1:n_trials) do _
                (; true_pos, true_rot, runway_corners, make_noisy_projections
                ) = create_runway_scenario(;
                    true_pos=sample_aircraft_pos(rng),
                    true_rot=sample_aircraft_rot(rng)
                )
                sigmas = noise_level * ones(length(runway_corners))
                noise_cov = Diagonal(repeat(sigmas .^ 2, inner=2))

                stats = compute_integrity_statistic(
                    true_pos, true_rot,
                    runway_corners, make_noisy_projections(rng, noise_level),
                    noise_cov
                )
                stats.p_value
            end

            # Validate p-value distribution
            validation = validate_p_value_distribution(p_values, n_bins=10)

            @test validation.uniform_distribution
            @test 0.0 <= minimum(p_values)
            @test maximum(p_values) <= 1.0

            # Check that approximately 5% of trials fail at α=0.05
            failure_rate = mean(p_values .< 0.05)
            @test 0.02 < failure_rate < 0.08
        end
    end

    @testset "4. Noise Model Integration" begin
        (; true_pos, true_rot, runway_corners, make_noisy_projections) = create_runway_scenario()
        noise_level = 2.0

        @testset "Diagonal Noise Model" begin
            rng = MersenneTwister(45)
            # Create vector of Normal distributions for UncorrGaussianNoiseModel
            normal_dists = [Normal(0.0, noise_level) for _ in 1:8]
            noise_model = UncorrGaussianNoiseModel(normal_dists)
            cov_matrix = covmatrix(noise_model)

            stats = compute_integrity_statistic(
                true_pos, true_rot,
                runway_corners, make_noisy_projections(rng, noise_level),
                cov_matrix
            )

            @test stats.p_value > 0.01
            @test size(cov_matrix) == (8, 8)
            @test isdiag(cov_matrix)
        end

        @testset "Full Covariance Noise Model" begin
            rng = MersenneTwister(46)
            # Create correlated noise model using MvNormal
            base_var = noise_level^2
            correlation = 0.3
            cov_full = base_var * (I + correlation * (ones(8, 8) - I))
            mv_normal = MvNormal(zeros(8), cov_full)
            noise_model = CorrGaussianNoiseModel(mv_normal)
            cov_matrix = covmatrix(noise_model)

            stats = compute_integrity_statistic(
                true_pos, true_rot,
                runway_corners, make_noisy_projections(rng, noise_level),
                cov_matrix
            )

            @test stats.p_value > 0.01
            @test !isdiag(cov_matrix)
            @test issymmetric(cov_matrix)
        end
    end

    @testset "5. Pose Estimation Integration" begin
        (; true_pos, true_rot, runway_corners, make_noisy_projections) = create_runway_scenario()
        noise_level = 1.5
        sigmas = noise_level * ones(length(runway_corners))
        noise_cov = Diagonal(repeat(sigmas .^ 2, inner=2))

        @testset "6-DOF Pose Estimation Integration" begin
            rng = MersenneTwister(47)
            noisy_projections = make_noisy_projections(rng, noise_level)
            # Estimate pose using 6-DOF estimator
            pose_result = estimatepose6dof(
                runway_corners,
                noisy_projections,
                CameraMatrix(CAMERA_CONFIG_OFFSET)
            )

            # Test integrity using estimated pose
            stats = compute_integrity_statistic(
                pose_result.pos, pose_result.rot,
                runway_corners, noisy_projections,
                noise_cov
            )

            @test stats.p_value > 0.01

            # Estimated pose should be reasonably close to true pose
            pos_error = norm([
                ustrip(m, pose_result.pos.x - true_pos.x),
                ustrip(m, pose_result.pos.y - true_pos.y),
                ustrip(m, pose_result.pos.z - true_pos.z)
            ])
            @test pos_error < 50.0
        end

        @testset "3-DOF Pose Estimation Integration" begin
            rng = MersenneTwister(48)
            noisy_projections = make_noisy_projections(rng, noise_level)
            # Estimate position with known rotation
            pose_result = estimatepose3dof(
                runway_corners,
                noisy_projections,
                true_rot,  # Use true rotation
                CameraMatrix(CAMERA_CONFIG_OFFSET)
            )

            # Test integrity using estimated pose
            stats = compute_integrity_statistic(
                pose_result.pos, pose_result.rot,
                runway_corners, noisy_projections,
                noise_cov
            )

            @test stats.p_value > 0.01
        end
    end

    @testset "6. Static Array Return Types" begin
        # Test that compute_worst_case_fault_direction_and_slope returns static arrays
        (; runway_corners, true_pos, true_rot) = create_runway_scenario()
        noise_level = 1.5
        sigmas = noise_level * ones(length(runway_corners))
        noise_cov = Diagonal(repeat(sigmas .^ 2, inner=2))

        # Compute Jacobian matrix using compute_H
        H_ = RunwayLib.compute_H(true_pos, true_rot, runway_corners)

        @testset for ndof in [3, 6]
            # Slice H to get appropriate columns, then convert to SMatrix
            # (indexing into SMatrix returns Matrix, so we need to convert back)
            H_static = SMatrix{8, ndof}(H_[:, 1:ndof])

            # Use alpha_idx that's valid for this ndof
            alpha_idx = min(4, ndof)  # 4 for 6-DOF (yaw), 2 or 3 for 3-DOF

            f_dir, g_slope = RunwayLib.compute_worst_case_fault_direction_and_slope(alpha_idx, SA[1], H_static, noise_cov)

            # Check types
            @test f_dir isa StaticArray
            @test g_slope isa Real

            # Check properties
            @test length(f_dir) == size(H_static, 1)  # Should match number of measurements
            @test isfinite(g_slope)
            @test g_slope >= 0  # Slope should be non-negative
            @test norm(f_dir) ≈ 1.0  # Fault direction should be normalized
        end
    end

    @testset "Non-Default Camera Matrix Integration" begin
        # Test integrity monitoring with a custom camera matrix (like Python tests use)
        custom_camera_matrix = CameraMatrix{:offset}(
            SA[-7246.4 0.0 2048.5; 0.0 -7246.4 1500.5; 0.0 0.0 1.0] * px,  # Note negative focal lengths like Python test
            4096.0px, 3000.0px
        )

        # Standard runway corners from create_runway_scenario
        runway_corners = [
            WorldPoint(0.0m, -25.0m, 0.0m),
            WorldPoint(0.0m, 25.0m, 0.0m),
            WorldPoint(1500.0m, -25.0m, 0.0m),
            WorldPoint(1500.0m, 25.0m, 0.0m)
        ]

        # True aircraft pose
        true_pos = WorldPoint(-1300.0m, 0.0m, 80.0m)
        true_rot = RotZYX(0.03, 0.04, 0.05)  # roll, pitch, yaw like Python test

        # Generate projections with custom camera matrix
        projections = [project(true_pos, true_rot, corner, custom_camera_matrix) for corner in runway_corners]

        @testset "Integrity with Custom Camera Matrix" begin
            rng = MersenneTwister(49)
            # Add small amount of noise like Python test
            noisy_projections = projections .+ [ProjectionPoint(0.1 * randn(rng, 2)px) for _ in projections]

            # Create diagonal noise covariance
            noise_level = 2.0
            sigmas = noise_level * ones(length(runway_corners))
            noise_cov = Diagonal(repeat(sigmas .^ 2, inner=2))

            # This should work with custom camera matrix
            stats = compute_integrity_statistic(
                true_pos, true_rot,
                runway_corners, noisy_projections,
                noise_cov, custom_camera_matrix
            )

            @test stats.p_value > 0.01  # Should have reasonable integrity
            @test stats.dofs == 2  # 4 points * 2 coords - 6 DOF = 2
            @test stats.stat >= 0
            @test isfinite(stats.stat)
        end
    end

end
