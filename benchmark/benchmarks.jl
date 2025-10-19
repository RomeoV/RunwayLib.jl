using BenchmarkTools
using RunwayLib
import RunwayLib: PointFeatures, LineFeatures, makecache, Line
using Rotations
using Unitful, Unitful.DefaultSymbols
using StaticArrays
using Random
using LinearAlgebra

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

# Define runway lines (4 lines from the corners)
const RUNWAY_LINES = SA[
    (RUNWAY_CORNERS[1], RUNWAY_CORNERS[3]),  # left edge
    (RUNWAY_CORNERS[2], RUNWAY_CORNERS[4]),  # right edge
    (RUNWAY_CORNERS[1], RUNWAY_CORNERS[2]),  # near edge
    (RUNWAY_CORNERS[3], RUNWAY_CORNERS[4]),  # far edge
]

# Ground truth airplane poses for benchmarks
const TRUE_POS = WorldPoint(-2500.0m, 10.0m, 150.0m)
const TRUE_ROT = RotZYX(roll=0.0, pitch=0.0, yaw=-0.0)

# Camera configurations to test
const CAMERA_CONFIGS = [
    (CAMERA_CONFIG_OFFSET, "CameraConfig :offset"),
    (CameraMatrix(CAMERA_CONFIG_OFFSET), "CameraMatrix: Offset"),
]

# Noise levels (in pixels)
const NOISE_LEVELS = [1.0, 3.0, 5.0]

function generate_projections_with_noise(noise_px::Float64, camera_config)
    # Generate perfect projections
    perfect_projections = [project(TRUE_POS, TRUE_ROT, corner, camera_config) for corner in RUNWAY_CORNERS]

    noisy_projections = [
        proj + ProjectionPoint(noise_px * randn(2) * px)
        for proj in perfect_projections
    ]
    return noisy_projections
end

function generate_observed_lines(noise_r, noise_theta, camera_config)
    # Generate perfect line observations
    observed_lines = map(RUNWAY_LINES) do (p1, p2)
        proj1 = project(TRUE_POS, TRUE_ROT, p1, camera_config)
        proj2 = project(TRUE_POS, TRUE_ROT, p2, camera_config)
        line = getline(proj1, proj2)
        Line(line.r + noise_r * randn(), line.theta + noise_theta * randn())
    end
    return observed_lines
end

# Function to generate random initial guesses with Gaussian errors
function generate_random_initial_guess()
    # Position errors: (500m, 50m, 20m) Gaussian noise
    pos_noise = SA[500.0m, 50.0m, 20.0m]
    pos_guess = SA[
        TRUE_POS.x+pos_noise[1]*randn(),
        TRUE_POS.y+pos_noise[2]*randn(),
        TRUE_POS.z+pos_noise[3]*randn(),
    ]

    # Rotation errors: (0.1rad, 0.1rad, 0.1rad) Gaussian noise
    rot_noise = 0.1  # rad
    rot_guess = SA[
        TRUE_ROT.theta1+rot_noise*randn(),
        TRUE_ROT.theta2+rot_noise*randn(),
        TRUE_ROT.theta3+rot_noise*randn(),
    ]rad

    return pos_guess, rot_guess
end

# Function for generating initial guesses with (100, 30, 10)m errors
function generate_smaller_initial_guess()
    # Position errors: (100m, 30m, 10m) Gaussian noise
    pos_noise = SA[100.0m, 10.0m, 10.0m]
    pos_guess = SA[
        TRUE_POS.x+pos_noise[1]*randn(),
        TRUE_POS.y+pos_noise[2]*randn(),
        TRUE_POS.z+pos_noise[3]*randn(),
    ]

    # Rotation errors: (0.1rad, 0.1rad, 0.1rad) Gaussian noise
    rot_noise = 0.01  # rad
    rot_guess = SA[
        TRUE_ROT.theta1+rot_noise*randn(),
        TRUE_ROT.theta2+rot_noise*randn(),
        TRUE_ROT.theta3+rot_noise*randn(),
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
                initial_guess_pos=pos_guess,
                initial_guess_rot=rot_guess
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
                initial_guess_pos=pos_guess
            )
        end setup = (Random.seed!(42))
    end
end

# 6DOF with Lines and Preallocated Cache Benchmarks
SUITE["6DOF+Lines"] = BenchmarkGroup()
for (camera_config, config_name) in CAMERA_CONFIGS
    SUITE["6DOF+Lines"][config_name] = BenchmarkGroup()
    for noise_level in NOISE_LEVELS
        noise_name = "$(Int(noise_level))px"

        # Preallocate features and cache
        projections_const = [project(TRUE_POS, TRUE_ROT, corner, camera_config) for corner in RUNWAY_CORNERS]
        observed_lines_const = generate_observed_lines(noise_level, deg2rad(noise_level), camera_config)

        # point_noise = SMatrix{8,8}(diagm(fill(noise_level^2, 8)))
        point_noise = Diagonal(SVector{8}(noise_level^2 * ones(8)))
        # line_noise = SMatrix{12,12}(diagm(fill(3.0^2, 12)))
        line_noise = Diagonal(SVector{12}(repeat([1^2, 0.02^2, 0.02^2], outer=4)))

        point_features_const = PointFeatures(
            collect(RUNWAY_CORNERS), projections_const,
            camera_config, point_noise
        )
        line_features_const = LineFeatures(
            collect(RUNWAY_LINES), observed_lines_const,
            camera_config, line_noise
        )
        # line_features_const = RunwayLib.NO_LINES

        # Create initial cache
        pos_guess_init, rot_guess_init = generate_smaller_initial_guess()
        u₀_init = vcat(ustrip.(m, pos_guess_init), ustrip.(rad, rot_guess_init))
        ps_const = PoseOptimizationParams6DOF(point_features_const, line_features_const)
        cache_init = RunwayLib.makecache(u₀_init, ps_const)

        SUITE["6DOF+Lines (cached)"][config_name][noise_name] = @benchmarkable begin
            # Run estimation with preallocated cache
            estimatepose6dof(
                point_features, line_features;
                initial_guess_pos=pos_guess,
                initial_guess_rot=rot_guess,
                cache=$cache_init,
            )
        end setup = begin
            # Generate noisy projections
            projections = generate_projections_with_noise($noise_level, $camera_config)
            observed_lines = generate_observed_lines(1.0px, deg2rad(1)rad, camera_config)

            # Update point features with noisy observations
            point_features = PointFeatures(
                collect($RUNWAY_CORNERS), projections,
                $camera_config, $point_noise
            )
            line_features = LineFeatures(
                collect(RUNWAY_LINES), observed_lines,
                $camera_config, $line_noise
            )

            # Generate new initial guess
            pos_guess, rot_guess = generate_smaller_initial_guess()
        end
    end
end
