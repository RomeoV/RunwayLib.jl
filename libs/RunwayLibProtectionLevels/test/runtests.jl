using Test
using RunwayLib
using RunwayLibProtectionLevels
using Distributions
using LinearAlgebra
using Rotations
using Random
using Unitful
using Unitful.DefaultSymbols
using StaticArrays
import RunwayLib: px
using JET: @test_opt

# =============================================================================
# Helpers
# =============================================================================

sample_aircraft_pos(rng) = let
    pos = rand(rng, MvNormal([-3000.0, 0, 300], Diagonal([1000.0, 200, 100] .^ 2)))
    pos = clamp.(pos, [-Inf, -Inf, 100], [-1000, Inf, Inf])
    WorldPoint(pos) * m
end

sample_aircraft_rot(rng) = RotZYX((deg2rad(5) * randn(rng, 3))...)

function eval_perturbation(Δy, obs_flat, σ_val, world_pts, cam_rot, noise_cov)
    perturbed_flat = obs_flat + Δy * σ_val
    n = div(length(perturbed_flat), 2)
    perturbed_obs = [ProjectionPoint(perturbed_flat[2i-1] * px, perturbed_flat[2i] * px) for i in 1:n]
    pose = estimatepose3dof(PointFeatures(world_pts, perturbed_obs), NO_LINES, cam_rot)
    sr = compute_integrity_statistic(pose.pos, pose.rot, world_pts, perturbed_obs, noise_cov)
    return (; pose, sr, norm2=sum(Δy .^ 2))
end

# =============================================================================
# Tests
# =============================================================================

@testset "RunwayLibProtectionLevels" begin

    runway_corners = [
        WorldPoint(0.0m, -25.0m, 0.0m),
        WorldPoint(0.0m, 25.0m, 0.0m),
        WorldPoint(1500.0m, -25.0m, 0.0m),
        WorldPoint(1500.0m, 25.0m, 0.0m),
    ]
    px_std = sqrt(2.0)
    noise_cov = Diagonal(px_std^2 * ones(8))

    @testset "Constraint satisfaction and optimality" begin
        rng = MersenneTwister(42)
        n_scenarios = 3

        for scenario_idx in 1:n_scenarios
            pos = sample_aircraft_pos(rng)
            rot = sample_aircraft_rot(rng)
            clean = [project(pos, rot, c) for c in runway_corners]
            noisy = clean .+ [ProjectionPoint(px_std * randn(rng, 2)px) for _ in clean]

            obs_flat = reduce(vcat, [SVector(ustrip(px, p.x), ustrip(px, p.y)) for p in noisy])
            σ_val = px_std

            for alpha_idx in 1:3
                @testset "scenario=$scenario_idx α=$alpha_idx" begin
                    result = compute_zero_fault_protection_level_grad(
                        runway_corners, noisy, noise_cov, rot;
                        alpha_idx, prob=0.01,
                    )

                    for (label, r) in [("lo", result.lo), ("hi", result.hi)]
                        @testset "$label" begin
                            # Test A: Constraint satisfaction at solution
                            @test r.protection_level ≥ 0
                            @test r.feasible

                            ev = eval_perturbation(r.Δy, obs_flat, σ_val, runway_corners, rot, noise_cov)
                            # Tolerance accounts for pose solver discrepancy (IFT vs cached estimatepose3dof)
                            @test ev.sr.stat ≤ result.stat_ref + 0.01
                            @test ev.norm2 ≤ result.chi2_bound + 1e-4

                            # Test B: 5% scaling violates constraints
                            Δy_scaled = 1.05 * r.Δy
                            ev_scaled = eval_perturbation(Δy_scaled, obs_flat, σ_val, runway_corners, rot, noise_cov)
                            @test ev_scaled.sr.stat > result.stat_ref ||
                                  sum(Δy_scaled .^ 2) > result.chi2_bound

                            # Test C: Probability bound (chi² check)
                            # At solution, p_value should be > prob (constraints not violated)
                            @test ev.sr.p_value > 0.01
                        end
                    end

                    # Test D: Random noise vectors can't beat the PL
                    @testset "random_dominance" begin
                        n_random = 100
                        for _ in 1:n_random
                            # Random Δy on the chi² sphere
                            Δy_rand = randn(rng, length(obs_flat))
                            Δy_rand *= sqrt(result.chi2_bound) / norm(Δy_rand) * rand(rng)

                            ev = eval_perturbation(Δy_rand, obs_flat, σ_val, runway_corners, rot, noise_cov)

                            # Skip if infeasible
                            (ev.sr.stat > result.stat_ref + 1e-4 || ev.norm2 > result.chi2_bound + 1e-4) && continue

                            # If feasible, position error should not exceed PL in either direction
                            pos_err = ustrip(m, ev.pose.pos[alpha_idx]) - ustrip(m, result.pos_ref[alpha_idx])
                            @test pos_err ≤ result.hi.protection_level + 0.5  # small tolerance for Ipopt approx
                            @test pos_err ≥ result.lo.protection_level - 0.5
                        end
                    end
                end
            end
        end
    end

    @testset "NelderMead vs Ipopt consistency" begin
        rng = MersenneTwister(123)
        pos = sample_aircraft_pos(rng)
        rot = sample_aircraft_rot(rng)
        clean = [project(pos, rot, c) for c in runway_corners]
        noisy = clean .+ [ProjectionPoint(px_std * randn(rng, 2)px) for _ in clean]

        for alpha_idx in 1:3
            @testset "α=$alpha_idx" begin
                r_ipopt = compute_zero_fault_protection_level_grad(
                    runway_corners, noisy, noise_cov, rot; alpha_idx, prob=0.01)
                r_nm = compute_zero_fault_protection_level(
                    runway_corners, noisy, noise_cov, rot; alpha_idx, prob=0.01)

                for dir in (:lo, :hi)
                    pl_ipopt = getfield(r_ipopt, dir).protection_level
                    pl_nm = getfield(r_nm, dir).protection_level
                    # Within 3% relative tolerance (or 0.1m absolute for small PLs)
                    @test isapprox(pl_ipopt, pl_nm; rtol=0.03, atol=0.1) ||
                          abs(pl_ipopt) > abs(pl_nm)  # Ipopt finding larger PL is also fine
                end
            end
        end
    end

    @testset "JET type stability" begin
        pos = WorldPoint(-800.0m, 5.0m, 120.0m)
        rot = RotZYX(0.02, 0.05, 0.01)
        corners = SA[
            WorldPoint(0.0m, -25.0m, 0.0m),
            WorldPoint(0.0m, 25.0m, 0.0m),
            WorldPoint(1500.0m, -25.0m, 0.0m),
            WorldPoint(1500.0m, 25.0m, 0.0m),
        ]

        clean = SVector([project(pos, rot, c) for c in corners])
        pf = PointFeatures(corners, clean)

        obs_flat = RunwayLibProtectionLevels._obs_to_flat(clean)
        pos_m = SVector(ustrip.(m, pos)...)
        Linv = pf.Linv
        cov = pf.cov

        @testset "helpers" begin
            @test_opt stacktrace_types_limit = 3 RunwayLibProtectionLevels._optvar2nominal_3dof(SA[1.0, 2.0, 3.0])
            @test_opt stacktrace_types_limit = 3 RunwayLibProtectionLevels._pos_to_world(SA[1.0, 2.0, 3.0])
            @test_opt stacktrace_types_limit = 3 RunwayLibProtectionLevels._obs_to_points(collect(obs_flat), Val(4))
        end

        @testset "pose residuals" begin
            residual_fn = RunwayLibProtectionLevels._make_pose_residuals(rot, corners, Linv, Val(4))
            u0 = collect(RunwayLibProtectionLevels._nominal2optvar_3dof(pos_m))
            @test_opt stacktrace_types_limit = 3 residual_fn(u0, collect(obs_flat))
        end

        @testset "stat computation" begin
            @test_opt stacktrace_types_limit = 3 RunwayLibProtectionLevels._compute_stat(
                pos_m, collect(obs_flat), corners, rot, cov, Linv, Val(4))
        end
    end
end
