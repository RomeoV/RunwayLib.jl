using BenchmarkTools
using RunwayLib
using Rotations
using Unitful, Unitful.DefaultSymbols
using StaticArrays
using Random

const SUITE = BenchmarkGroup()

# Set seed for reproducible benchmarks
Random.seed!(42)

# Define standard runway corners (4 points forming a rectangle)
const RUNWAY_CORNERS = SA[
    WorldPoint(0.0m, 25.0m, 0.0m),      # near left
    WorldPoint(0.0m, -25.0m, 0.0m),     # near right
    WorldPoint(3000.0m, 25.0m, 0.0m),   # far left
    WorldPoint(3000.0m, -25.0m, 0.0m),  # far right
]

# Ground truth airplane poses for benchmarks
const TRUE_POS = WorldPoint(-2500.0m, 10.0m, 150.0m)
const TRUE_ROT = RotZYX(roll = 0.0, pitch = 0.0, yaw = -0.0)

# Camera configurations to test
const CAMERA_CONFIGS = [
    (CAMERA_CONFIG_OFFSET, "Offset"),
]

# Noise levels (in pixels)
const NOISE_LEVELS = [0.0, 1.0, 3.0]

function generate_projections_with_noise(noise_px::Float64, camera_config)
    # Generate perfect projections
    perfect_projections = [project(TRUE_POS, TRUE_ROT, corner, camera_config) for corner in RUNWAY_CORNERS]

    noisy_projections = [
        proj + ProjectionPoint(noise_px * randn(2) * px)
            for proj in perfect_projections
    ]
    return noisy_projections
end

# Function to generate random initial guesses with Gaussian errors
function generate_random_initial_guess()
    # Position errors: (500m, 50m, 20m) Gaussian noise
    pos_noise = SA[500.0m, 50.0m, 20.0m]
    pos_guess = SA[
        TRUE_POS.x + pos_noise[1] * randn(),
        TRUE_POS.y + pos_noise[2] * randn(),
        TRUE_POS.z + pos_noise[3] * randn(),
    ]

    # Rotation errors: (0.1rad, 0.1rad, 0.1rad) Gaussian noise
    rot_noise = 0.1  # rad
    rot_guess = SA[
        TRUE_ROT.theta1 + rot_noise * randn(),
        TRUE_ROT.theta2 + rot_noise * randn(),
        TRUE_ROT.theta3 + rot_noise * randn(),
    ]rad

    return pos_guess, rot_guess
end

# 6DOF Pose Estimation Benchmarks
SUITE["6DOF"] = BenchmarkGroup()
for (camera_config, config_name) in CAMERA_CONFIGS
    SUITE["6DOF"][config_name] = BenchmarkGroup()
    for noise_level in NOISE_LEVELS
        noise_name = "$(Int(noise_level))px"
        SUITE["6DOF"][config_name][noise_name] = @benchmarkable begin
            projections = generate_projections_with_noise($noise_level, $camera_config)
            pos_guess, rot_guess = generate_random_initial_guess()
            estimatepose6dof(
                $RUNWAY_CORNERS,
                projections,
                $camera_config;
                initial_guess_pos = pos_guess,
                initial_guess_rot = rot_guess
            )
        end setup = (Random.seed!(42))
    end
end

# 3DOF Pose Estimation Benchmarks
SUITE["3DOF"] = BenchmarkGroup()
for (camera_config, config_name) in CAMERA_CONFIGS
    SUITE["3DOF"][config_name] = BenchmarkGroup()
    for noise_level in NOISE_LEVELS
        noise_name = "$(Int(noise_level))px"
        SUITE["3DOF"][config_name][noise_name] = @benchmarkable begin
            projections = generate_projections_with_noise($noise_level, $camera_config)
            pos_guess, _ = generate_random_initial_guess()
            estimatepose3dof(
                $RUNWAY_CORNERS,
                projections,
                $TRUE_ROT,
                $camera_config;
                initial_guess_pos = pos_guess
            )
        end setup = (Random.seed!(42))
    end
end
