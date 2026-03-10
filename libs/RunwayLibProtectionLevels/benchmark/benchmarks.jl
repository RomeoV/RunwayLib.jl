using RunwayLibProtectionLevels
using RunwayLib
using Chairmarks
using Unitful, Unitful.DefaultSymbols
using StaticArrays
using Rotations: RotZYX
using LinearAlgebra
using Random

# --- Setup: representative scenario ---
rng = MersenneTwister(42)
runway_corners = SA[
    WorldPoint(0.0m, -25.0m, 0.0m),
    WorldPoint(0.0m, 25.0m, 0.0m),
    WorldPoint(1500.0m, -25.0m, 0.0m),
    WorldPoint(1500.0m, 25.0m, 0.0m),
]
px_std = sqrt(2.0)
noise_cov = Diagonal(px_std^2 * ones(8))
cam_pos = WorldPoint(-2500.0m, 10.0m, 250.0m)
cam_rot = RotZYX(0.02, 0.05, 0.01)
clean_obs = [project(cam_pos, cam_rot, c) for c in runway_corners]
noisy_obs = clean_obs .+ [ProjectionPoint(px_std * randn(rng, 2)RunwayLib.px) for _ in clean_obs]

# Warmup
compute_zero_fault_protection_level_grad(runway_corners, noisy_obs, noise_cov, cam_rot; alpha_idx=1, prob=0.01)

# --- Benchmarks ---
println("PL both directions (along-track):")
display(@be compute_zero_fault_protection_level_grad(
    runway_corners, noisy_obs, noise_cov, cam_rot;
    alpha_idx=1, direction=0, prob=0.01,
) evals=1 samples=20 seconds=60)

println("\nPL single direction (along-track, hi):")
display(@be compute_zero_fault_protection_level_grad(
    runway_corners, noisy_obs, noise_cov, cam_rot;
    alpha_idx=1, direction=1, prob=0.01,
) evals=1 samples=20 seconds=60)

println("\nPL both directions (cross-track):")
display(@be compute_zero_fault_protection_level_grad(
    runway_corners, noisy_obs, noise_cov, cam_rot;
    alpha_idx=2, direction=0, prob=0.01,
) evals=1 samples=20 seconds=60)
