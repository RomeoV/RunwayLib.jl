# Enhanced C API for RunwayLib pose estimation
# Provides C-callable functions for external language bindings

using StaticArrays
using Rotations
using Unitful, Unitful.DefaultSymbols
using CEnum
using Distributions: Normal, MvNormal
using LinearAlgebra: isposdef
using ProbabilisticParameterEstimators: CorrGaussianNoiseModel, UncorrGaussianNoiseModel, NoiseModel
using LightSumTypes: @sumtype, variant

# Type aliases for C interop
# These use the same memory layout as the existing parametric structs
const WorldPointF64 = WorldPoint{Float64}
const ProjectionPointF64 = ProjectionPoint{Float64,:offset}

# struct Rotation_C
#     yaw::Float64
#     pitch::Float64
#     roll::Float64
# end
const RotYPRF64 = SVector{3,Float64}

@kwdef struct PoseEstimate_C
    position::WorldPointF64 = WorldPointF64(0.0, 0.0, 0.0)
    rotation::RotYPRF64 = RotYPRF64(0.0, 0.0, 0.0)
    residual_norm::Float64 = 0.0
    converged::Cint = 0
end

# Type alias for integrity monitoring results - matches NamedTuple layout with C-compatible types
const IntegrityResult_C = @NamedTuple{stat::Float64, p_value::Float64, dofs::Cint, residual_norm::Float64}

struct CameraMatrix_C
    matrix::SMatrix{3,3,Float64,9}  # 3x3 camera matrix (unitless)
    image_width::Float64           # Image width in pixels (unitless)
    image_height::Float64          # Image height in pixels (unitless)
    coordinate_system::Cint        # 0 for centered, 1 for offset
end

@cenum POSEEST_ERROR::Cint begin
    POSEEST_SUCCESS = 0
    POSEEST_ERROR_INVALID_INPUT = -1
    POSEEST_ERROR_BEHIND_CAMERA = -2
    POSEEST_ERROR_NO_CONVERGENCE = -3
    POSEEST_ERROR_INSUFFICIENT_POINTS = -4
end

# Global variable to track library initialization
const LIBRARY_INITIALIZED = Ref(false)

# Remove CAMERA_CONFIG_C enum - now using CameraMatrix_C directly

module CovarianceType
using CEnum
@cenum COVARIANCE_TYPE_C::Cint begin
    COV_DEFAULT = 0         # Use default noise model (pointer can be null)
    COV_SCALAR = 1          # Single noise value for all keypoints/directions
    COV_DIAGONAL_FULL = 2   # Diagonal matrix (length = 2*n_keypoints)
    COV_BLOCK_DIAGONAL = 3  # 2x2 matrix per keypoint (length = 4*n_keypoints)
    COV_FULL_MATRIX = 4     # Full covariance matrix (length = 4*n_keypoints^2)
end
end
using .CovarianceType: COVARIANCE_TYPE_C, COV_DEFAULT, COV_SCALAR, COV_DIAGONAL_FULL, COV_BLOCK_DIAGONAL, COV_FULL_MATRIX

const NORMAL_T = Normal{Float64}
const MVNORMAL_T = typeof(MvNormal(rand(2), Matrix(1.0 * I(2))))
# Define sum type for all possible noise models
@sumtype NoiseModelVariant(
    UncorrGaussianNoiseModel{NORMAL_T,Vector{NORMAL_T}},
    UncorrGaussianNoiseModel{MVNORMAL_T,Vector{MVNORMAL_T}},
    CorrGaussianNoiseModel{MVNORMAL_T}
)

function get_camera_matrix_from_c(camera_matrix_c::CameraMatrix_C)
    # Only :offset coordinate system supported
    # Note: coordinate_system field ignored - always use :offset

    # C arrays are row-major, but Julia SMatrix is column-major, so transpose
    matrix_transposed = transpose(camera_matrix_c.matrix)

    # Create CameraMatrix with proper units
    matrix_with_units = matrix_transposed * px
    width_with_units = camera_matrix_c.image_width * px
    height_with_units = camera_matrix_c.image_height * px

    # Create CameraMatrix with :offset coordinates
    return CameraMatrix{:offset}(matrix_with_units, width_with_units, height_with_units)
end

function parse_covariance_data(covariance_type::COVARIANCE_TYPE_C, covariance_data::Ptr{Cdouble}, num_points::Integer)
    """
    Parse covariance data from C pointer based on covariance type enum.
    Returns a NoiseModel that can be used with the optimization infrastructure.
    """
    if covariance_type == COV_DEFAULT
        # Return default noise model (pointer can be null in this case)
        return _defaultnoisemodel(1:num_points) |> NoiseModelVariant
    end

    # For all other types, we need valid covariance data
    if covariance_data == C_NULL
        throw(ArgumentError("Covariance data pointer cannot be null for covariance type: $covariance_type"))
    end

    if covariance_type == COV_SCALAR
        let
            # Single noise standard deviation for all measurements
            noise_std = unsafe_load(covariance_data, 1)
            if noise_std <= 0
                throw(ArgumentError("Noise standard deviation must be positive"))
            end
            distributions = [Normal(0.0, noise_std) for _ in 1:(2*num_points)]
            return UncorrGaussianNoiseModel(distributions) |> NoiseModelVariant
        end
    elseif covariance_type == COV_DIAGONAL_FULL
        let
            # Diagonal covariance matrix: variances for each measurement
            variances = unsafe_wrap(Array, covariance_data, 2 * num_points)
            if any(variances .<= 0)
                throw(ArgumentError("All variances must be positive"))
            end
            distributions = [Normal(0.0, sqrt(var)) for var in variances]
            return UncorrGaussianNoiseModel(distributions) |> NoiseModelVariant
        end
    elseif covariance_type == COV_BLOCK_DIAGONAL
        let
            # 2x2 covariance matrix for each keypoint
            data_length = 4 * num_points
            cov_data = unsafe_wrap(Array, covariance_data, data_length)

            # Create individual 2x2 covariance matrices for each keypoint
            distributions = Vector{MVNORMAL_T}(undef, num_points)
            for i in 1:num_points
                idx = (1:4) .+ (i - 1) * 4
                cov_2x2 = reshape(cov_data[idx], 2, 2)

                if !isposdef(cov_2x2)
                    throw(ArgumentError("Covariance matrix for keypoint $i must be positive definite"))
                end

                distributions[i] = MvNormal(zeros(2), cov_2x2)
            end
            return UncorrGaussianNoiseModel(distributions) |> NoiseModelVariant
        end
    elseif covariance_type == COV_FULL_MATRIX
        let
            # Full covariance matrix - use CorrGaussianNoiseModel for correlated observations
            matrix_size = 2 * num_points
            data_length = matrix_size * matrix_size
            cov_data = unsafe_wrap(Array, covariance_data, data_length)

            # Reconstruct matrix from row-major storage, but since symmetric don't worry
            cov_matrix = reshape(cov_data, matrix_size, matrix_size)

            if !isposdef(cov_matrix) || !issymmetric(cov_matrix)
                throw(ArgumentError("Full covariance matrix must be symmetric and positive definite"))
            end

            # Create a single MvNormal for all measurements (correlated case)
            corr_noise = MvNormal(zeros(matrix_size), cov_matrix)
            return CorrGaussianNoiseModel(corr_noise) |> NoiseModelVariant
        end
    else
        throw(ArgumentError("Invalid covariance type: $covariance_type"))
    end
end

# Error message function
function get_error_message_impl(error_code::POSEEST_ERROR)
    messages = Dict(
        POSEEST_SUCCESS => "Success",
        POSEEST_ERROR_INVALID_INPUT => "Invalid input parameters",
        POSEEST_ERROR_BEHIND_CAMERA => "Point is behind camera",
        POSEEST_ERROR_NO_CONVERGENCE => "Optimization did not converge",
        POSEEST_ERROR_INSUFFICIENT_POINTS => "Insufficient number of points"
    )
    return get(messages, error_code, "Unknown error")
end

# Store error messages in a global to ensure they persist
const ERROR_MESSAGES = Dict{Int,Ptr{UInt8}}()

# Static buffer for detailed error messages from exceptions
const LAST_ERROR_BUFFER = Vector{UInt8}(undef, 500)

function set_last_error(msg::String)
    # Simple AOT-friendly error formatting - just copy the string
    bytes = codeunits(msg)
    n = min(length(bytes), length(LAST_ERROR_BUFFER) - 1)
    copyto!(LAST_ERROR_BUFFER, 1, bytes, 1, n)
    LAST_ERROR_BUFFER[n + 1] = 0  # null terminator
end

# Type-specific error message formatters (AOT-friendly - using only fixed strings)
set_last_error(fn::String, ::ArgumentError) = set_last_error(fn * ": ArgumentError")
set_last_error(fn::String, ::BoundsError) = set_last_error(fn * ": BoundsError")
set_last_error(fn::String, ::Exception) = set_last_error(fn * ": Unknown error")

Base.@ccallable function get_error_message(error_code::Cint)::Ptr{UInt8}
    if !haskey(ERROR_MESSAGES, error_code)
        msg = get_error_message_impl(POSEEST_ERROR(error_code))
        ERROR_MESSAGES[error_code] = pointer(msg)
    end
    return ERROR_MESSAGES[error_code]
end

Base.@ccallable function get_last_error_detail()::Ptr{UInt8}
    return pointer(LAST_ERROR_BUFFER)
end

# 6DOF pose estimation with covariance specification and initial guess
Base.@ccallable function estimate_pose_6dof(
    runway_corners_::Ptr{WorldPointF64},
    projections_::Ptr{ProjectionPointF64},
    num_points::Cint,
    covariance_data::Ptr{Cdouble},
    covariance_type::COVARIANCE_TYPE_C,
    camera_matrix::Ptr{CameraMatrix_C},
    initial_guess_pos::Ptr{WorldPointF64},
    initial_guess_rot::Ptr{RotYPRF64},
    result::Ptr{PoseEstimate_C}
)::Cint
    try
        # Validate inputs
        if runway_corners_ == C_NULL || projections_ == C_NULL || result == C_NULL
            return POSEEST_ERROR_INVALID_INPUT
        end

        if num_points < 4
            return POSEEST_ERROR_INSUFFICIENT_POINTS
        end

        # Convert C arrays to Julia arrays
        runway_corners = unsafe_wrap(Array, runway_corners_, num_points) .* 1m
        projections = unsafe_wrap(Array, projections_, num_points) .* 1px

        # Parse covariance specification
        noise_model_variant = parse_covariance_data(covariance_type, covariance_data, num_points)

        # Validate camera matrix
        if camera_matrix == C_NULL
            return POSEEST_ERROR_INVALID_INPUT
        end

        # Load camera matrix and convert to CameraMatrix
        camera_matrix_c = unsafe_load(camera_matrix)
        cam_matrix = get_camera_matrix_from_c(camera_matrix_c)

        # Handle initial guess parameters
        if initial_guess_pos != C_NULL
            initial_pos_c = unsafe_load(initial_guess_pos)
            initial_pos = SA[initial_pos_c.x, initial_pos_c.y, initial_pos_c.z] * 1m
        else
            initial_pos = SA[-1000.0, 0.0, 100.0]m  # Default value
        end

        if initial_guess_rot != C_NULL
            initial_rot_c = unsafe_load(initial_guess_rot)
            initial_rot = initial_rot_c * 1rad
        else
            initial_rot = SA[0.0, 0.0, 0.0]rad  # Default value
        end

        # Perform pose estimation with custom noise model and initial guesses
        sol = estimatepose6dof(runway_corners, projections, cam_matrix, variant(noise_model_variant);
            # sol = estimatepose6dof(runway_corners, projections, cam_matrix, noise_model;
            initial_guess_pos=initial_pos, initial_guess_rot=initial_rot)

        # Convert result back to C struct
        result_c = PoseEstimate_C(
            sol.pos .|> _ustrip(m),
            Rotations.params(sol.rot),
            0.0,  # residual_norm - could be computed from solution
            1     # converged (assume success if no exception)
        )

        # Write result to output pointer
        unsafe_store!(result, result_c)

        return POSEEST_SUCCESS

    catch e
        # Only set generic error if no specific error was already set
        if LAST_ERROR_BUFFER[1] == 0
            set_last_error("estimate_pose_6dof", e)
        end
        if isa(e, BoundsError) || isa(e, ArgumentError)
            return POSEEST_ERROR_INVALID_INPUT
        else
            return POSEEST_ERROR_NO_CONVERGENCE
        end
    end
end

# 3DOF pose estimation with covariance specification and initial guess
Base.@ccallable function estimate_pose_3dof(
    runway_corners_::Ptr{WorldPointF64},
    projections_::Ptr{ProjectionPointF64},
    num_points::Cint,
    known_rotation::Ptr{RotYPRF64},
    covariance_data::Ptr{Cdouble},
    covariance_type::COVARIANCE_TYPE_C,
    camera_matrix::Ptr{CameraMatrix_C},
    initial_guess_pos::Ptr{WorldPointF64},
    result::Ptr{PoseEstimate_C}
)::Cint
    try
        # Validate inputs
        if runway_corners_ == C_NULL || projections_ == C_NULL || known_rotation == C_NULL || result == C_NULL
            return POSEEST_ERROR_INVALID_INPUT
        end

        if num_points < 3
            return POSEEST_ERROR_INSUFFICIENT_POINTS
        end

        # Convert C arrays to Julia arrays
        runway_corners = unsafe_wrap(Array, runway_corners_, num_points) .* 1m
        projections = unsafe_wrap(Array, projections_, num_points) .* 1px
        known_rot_c = unsafe_load(known_rotation)

        # Convert rotation to Julia type
        jl_rotation = RotZYX(known_rot_c...)

        # Parse covariance specification
        noise_model_variant = parse_covariance_data(covariance_type, covariance_data, num_points)

        # Validate camera matrix
        if camera_matrix == C_NULL
            return POSEEST_ERROR_INVALID_INPUT
        end

        # Load camera matrix and convert to CameraMatrix
        camera_matrix_c = unsafe_load(camera_matrix)
        cam_matrix = get_camera_matrix_from_c(camera_matrix_c)

        # Handle initial guess for position
        if initial_guess_pos != C_NULL
            initial_pos_c = unsafe_load(initial_guess_pos)
            initial_pos = SA[initial_pos_c.x, initial_pos_c.y, initial_pos_c.z] * 1m
        else
            initial_pos = SA[-1000.0, 0.0, 100.0]m  # Default value
        end

        # Perform pose estimation with custom noise model and initial guess
        sol = estimatepose3dof(runway_corners, projections, jl_rotation, cam_matrix, variant(noise_model_variant);
            initial_guess_pos=initial_pos)

        # Convert result back to C struct
        result_c = PoseEstimate_C(
            sol.pos .|> _ustrip(m),
            Rotations.params(jl_rotation),  # Use known rotation
            0.0,  # residual_norm
            1     # converged
        )

        # Write result to output pointer
        unsafe_store!(result, result_c)

        return POSEEST_SUCCESS

    catch e
        if isa(e, BoundsError) || isa(e, ArgumentError)
            set_last_error("estimate_pose_3dof", e)
            return POSEEST_ERROR_INVALID_INPUT
        else
            set_last_error("estimate_pose_3dof", e)
            return POSEEST_ERROR_NO_CONVERGENCE
        end
    end
end



# Point projection utility
Base.@ccallable function project_point(
    camera_position::Ptr{WorldPointF64},
    camera_rotation::Ptr{RotYPRF64},
    world_point::Ptr{WorldPointF64},
    camera_matrix::Ptr{CameraMatrix_C},
    result::Ptr{ProjectionPointF64}
)::Cint
    # Validate inputs
    if camera_position == C_NULL || camera_rotation == C_NULL || world_point == C_NULL || result == C_NULL
        return POSEEST_ERROR_INVALID_INPUT
    end

    # Load C structs
    cam_pos_c = unsafe_load(camera_position)
    cam_rot_c = unsafe_load(camera_rotation)
    world_pt_c = unsafe_load(world_point)

    # Convert to Julia types
    jl_cam_pos = cam_pos_c .* 1m
    jl_cam_rot = RotZYX(cam_rot_c[1], cam_rot_c[2], cam_rot_c[3])
    jl_world_pt = world_pt_c .* 1m

    # Validate camera matrix
    if camera_matrix == C_NULL
        return POSEEST_ERROR_INVALID_INPUT
    end

    # Load camera matrix and convert to CameraMatrix
    camera_matrix_c = unsafe_load(camera_matrix)
    cam_matrix = get_camera_matrix_from_c(camera_matrix_c)

    # Project point
    jl_projection = project(jl_cam_pos, jl_cam_rot, jl_world_pt, cam_matrix)

    # Convert result back to C struct
    result_c = jl_projection .|> _ustrip(px)

    # Write result to output pointer
    unsafe_store!(result, result_c)

    return POSEEST_SUCCESS
end

# Integrity monitoring function
Base.@ccallable function compute_integrity(
    camera_position::Ptr{WorldPointF64},
    camera_rotation::Ptr{RotYPRF64},
    runway_corners_::Ptr{WorldPointF64},
    projections_::Ptr{ProjectionPointF64},
    num_points::Cint,
    covariance_data::Ptr{Cdouble},
    covariance_type::COVARIANCE_TYPE_C,
    camera_matrix::Ptr{CameraMatrix_C},
    result::Ptr{IntegrityResult_C}
)::Cint
    try
        # Validate inputs
        if camera_position == C_NULL || camera_rotation == C_NULL ||
           runway_corners_ == C_NULL || projections_ == C_NULL || result == C_NULL
            return POSEEST_ERROR_INVALID_INPUT
        end

        if num_points < 4  # Need at least 4 points for 6-DOF integrity monitoring
            return POSEEST_ERROR_INSUFFICIENT_POINTS
        end

        # Convert camera pose from C to Julia types
        cam_pos_c = unsafe_load(camera_position)
        cam_rot_c = unsafe_load(camera_rotation)
        jl_cam_pos = cam_pos_c .* 1m
        jl_cam_rot = RotZYX(cam_rot_c[1], cam_rot_c[2], cam_rot_c[3])

        # Convert C arrays to Julia arrays
        runway_corners = unsafe_wrap(Array, runway_corners_, num_points) .* 1m
        projections = unsafe_wrap(Array, projections_, num_points) .* 1px

        # Parse covariance specification
        noise_cov_variant = parse_covariance_data(covariance_type, covariance_data, num_points)

        # Validate camera matrix
        if camera_matrix == C_NULL
            return POSEEST_ERROR_INVALID_INPUT
        end

        # Load camera matrix and convert to CameraMatrix
        camera_matrix_c = unsafe_load(camera_matrix)
        cam_matrix = get_camera_matrix_from_c(camera_matrix_c)

        # Compute integrity statistics
        integrity_result = compute_integrity_statistic(
            jl_cam_pos, jl_cam_rot,
            runway_corners, projections,
            variant(noise_cov_variant), cam_matrix
        )

        # Convert result to C-compatible NamedTuple (cast dofs to Cint)
        result_c = IntegrityResult_C((
            integrity_result.stat,
            integrity_result.p_value,
            Cint(integrity_result.dofs),
            integrity_result.residual_norm |> _ustrip(px)
        ))

        # Write result to output pointer
        unsafe_store!(result, result_c)

        return POSEEST_SUCCESS

    catch e
        if isa(e, BoundsError) || isa(e, ArgumentError)
            set_last_error("compute_integrity", e)
            return POSEEST_ERROR_INVALID_INPUT
        else
            set_last_error("compute_integrity", e)
            return POSEEST_ERROR_NO_CONVERGENCE
        end
    end
end
