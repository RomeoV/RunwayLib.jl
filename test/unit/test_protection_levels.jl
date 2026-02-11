using Test
using RunwayLib
using Distributions
using LinearAlgebra
using Rotations
using Random
using Unitful
using Unitful.DefaultSymbols
import RunwayLib: px, _ustrip
using RunwayLib.StaticArrays: SVector

sample_aircraft_pos() =
    let
        pos = rand(
            MvNormal([-3000.0, 0, 300], Diagonal([1000.0, 200, 100] .^ 2))
        )
        pos = clamp.(pos, [-Inf, -Inf, 100], [-1000, Inf, Inf])
        WorldPoint(pos) * m
    end
sample_aircraft_rot() = RotZYX((deg2rad(5) * randn(3))...)

function eval_perturbation_external(Δy, obs_flat, σ_val, world_pts, cam_rot, noise_cov)
    perturbed_flat = obs_flat + Δy * σ_val
    perturbed_obs = [ProjectionPoint(el * px) for el in eachcol(reshape(perturbed_flat, 2, :))]
    pose = estimatepose3dof(PointFeatures(world_pts, perturbed_obs), NO_LINES, cam_rot)
    sr = compute_integrity_statistic(pose.pos, pose.rot, world_pts, perturbed_obs, noise_cov)
    return (; pose, sr, norm2=sum(Δy .^ 2))
end

function check_single_direction(result, obs_flat, σ_val, runway_corners, rot)
    @test result.protection_level ≥ 0
    @test result.feasible

    ev = eval_perturbation_external(
        result.Δy, obs_flat, σ_val, runway_corners, rot, noise_cov
    )
    @test ev.sr.stat ≤ result.stat_ref + 1e-6
    @test ev.norm2 ≤ result.chi2_bound + 1e-6

    Δy_scaled = 1.05 * result.Δy
    ev_scaled = eval_perturbation_external(
        Δy_scaled, obs_flat, σ_val, runway_corners, rot, noise_cov
    )
    @test ev_scaled.sr.stat > result.stat_ref ||
          sum(Δy_scaled .^ 2) > result.chi2_bound
end

@testset "Zero-fault protection levels" begin
    runway_corners = [
        WorldPoint(0.0m, -25.0m, 0.0m),
        WorldPoint(0.0m, 25.0m, 0.0m),
        WorldPoint(1500.0m, -25.0m, 0.0m),
        WorldPoint(1500.0m, 25.0m, 0.0m),
    ]
    px_std = sqrt(2.0)
    noise_cov = Diagonal(px_std^2 * ones(8))

    Random.seed!(42)

    n_scenarios = 10
    n_obs_per_scenario = 10

    for scenario_idx in 1:n_scenarios
        pos = sample_aircraft_pos()
        rot = sample_aircraft_rot()
        clean = [project(pos, rot, c) for c in runway_corners]

        for obs_idx in 1:n_obs_per_scenario
            noisy = clean .+ [ProjectionPoint(px_std * randn(2)px) for _ in clean]

            obs_flat = reduce(vcat, [SVector(ustrip(px, p.x), ustrip(px, p.y)) for p in noisy])
            σ_val = px_std

            for alpha_idx in 1:3
                @testset "scenario=$scenario_idx obs=$obs_idx α=$alpha_idx" begin
                    # Test direction=0 (both)
                    result = compute_zero_fault_protection_level(
                        runway_corners, noisy, noise_cov, rot;
                        alpha_idx, prob=0.01,
                    )

                    @testset "lo (dir=-1)" begin
                        check_single_direction(
                            merge(result.lo, (; stat_ref=result.stat_ref, chi2_bound=result.chi2_bound)),
                            obs_flat, σ_val, runway_corners, rot,
                        )
                    end

                    @testset "hi (dir=+1)" begin
                        check_single_direction(
                            merge(result.hi, (; stat_ref=result.stat_ref, chi2_bound=result.chi2_bound)),
                            obs_flat, σ_val, runway_corners, rot,
                        )
                    end
                end
            end
        end
    end
end
