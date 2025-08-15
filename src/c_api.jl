# Enhanced C API for RunwayLib pose estimation
# Provides C-callable functions for external language bindings

using StaticArrays
using Rotations
using Unitful, Unitful.DefaultSymbols
using CEnum
using Distributions: Normal, MvNormal
using LinearAlgebra: isposdef
using ProbabilisticParameterEstimators: CorrGaussianNoiseModel

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

struct PoseEstimate_C
    position::WorldPointF64
    rotation::RotYPRF64
    residual_norm::Float64
    converged::Cint
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

@cenum(CAMERA_CONFIG_C, CAMERA_CONFIG_CENTERED_C = Cint(0), CAMERA_CONFIG_OFFSET_C = Cint(1))

@cenum COVARIANCE_TYPE_C::Cint begin
    COV_SCALAR = 0          # Single noise value for all keypoints/directions
    COV_DIAGONAL_FULL = 1   # Diagonal matrix (length = 2*n_keypoints)  
    COV_BLOCK_DIAGONAL = 2  # 2x2 matrix per keypoint (length = 4*n_keypoints)
    COV_FULL_MATRIX = 3     # Full covariance matrix (length = 4*n_keypoints^2)
end

function get_camera_config(config_type::CAMERA_CONFIG_C)
    if config_type == CAMERA_CONFIG_CENTERED_C
        return CAMERA_CONFIG_CENTERED
    elseif config_type == CAMERA_CONFIG_OFFSET_C
        return CAMERA_CONFIG_OFFSET
    else
        throw(ArgumentError("Invalid camera config type: $config_type"))
    end
end

function parse_covariance_data(covariance_data::Ptr{Cdouble}, covariance_type::COVARIANCE_TYPE_C, num_points::Int)
    """
    Parse covariance data from C pointer based on covariance type enum.
    Returns a NoiseModel that can be used with the optimization infrastructure.
    """
    if covariance_data == C_NULL
        # Return default noise model if no covariance data provided
        dummy_projections = [ProjectionPoint(0.0px, 0.0px) for _ in 1:num_points]
        return _defaultnoisemodel(dummy_projections)
    end
    
    if covariance_type == COV_SCALAR
        # Single noise standard deviation for all measurements
        noise_std = unsafe_load(covariance_data, 1)
        if noise_std <= 0
            throw(ArgumentError("Noise standard deviation must be positive"))
        end
        distributions = [Normal(0.0, noise_std) for _ in 1:(2*num_points)]
        return UncorrGaussianNoiseModel(distributions)
        
    elseif covariance_type == COV_DIAGONAL_FULL
        # Diagonal covariance matrix: variances for each measurement
        variances = unsafe_wrap(Array, covariance_data, 2*num_points)
        if any(variances .<= 0)
            throw(ArgumentError("All variances must be positive"))
        end
        distributions = [Normal(0.0, sqrt(var)) for var in variances]
        return UncorrGaussianNoiseModel(distributions)
        
    elseif covariance_type == COV_BLOCK_DIAGONAL
        # 2x2 covariance matrix for each keypoint
        data_length = 4 * num_points
        cov_data = unsafe_wrap(Array, covariance_data, data_length)
        
        distributions = MvNormal[]
        for i in 1:num_points
            # Extract 2x2 covariance matrix for this keypoint (stored row-major)
            base_idx = (i-1) * 4
            cov_2x2 = [cov_data[base_idx + 1] cov_data[base_idx + 2];
                       cov_data[base_idx + 3] cov_data[base_idx + 4]]
            
            # Check positive definite
            if !isposdef(cov_2x2)
                throw(ArgumentError("Covariance matrix for keypoint $i must be positive definite"))
            end
            
            push!(distributions, MvNormal(zeros(2), cov_2x2))
        end
        return UncorrGaussianNoiseModel(distributions)
        
    elseif covariance_type == COV_FULL_MATRIX
        # Full covariance matrix - use CorrGaussianNoiseModel for correlated observations
        matrix_size = 2 * num_points
        data_length = matrix_size * matrix_size
        cov_data = unsafe_wrap(Array, covariance_data, data_length)
        
        # Reconstruct matrix from row-major storage
        cov_matrix = reshape(cov_data, matrix_size, matrix_size)'  # Transpose for column-major
        
        if !isposdef(cov_matrix)
            throw(ArgumentError("Full covariance matrix must be positive definite"))
        end
        
        # Create a single MvNormal for all measurements (correlated case)
        corr_noise = MvNormal(zeros(matrix_size), cov_matrix)
        return CorrGaussianNoiseModel(corr_noise)
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

Base.@ccallable function get_error_message(error_code::Cint)::Ptr{UInt8}
    if !haskey(ERROR_MESSAGES, error_code)
        msg = get_error_message_impl(POSEEST_ERROR(error_code))
        ERROR_MESSAGES[error_code] = pointer(msg)
    end
    return ERROR_MESSAGES[error_code]
end

# Library initialization
Base.@ccallable function initialize_poseest_library(depot_path::Ptr{UInt8})::Cint
    try
        if depot_path != C_NULL
            path_str = unsafe_string(depot_path)
            ENV["JULIA_DEPOT_PATH"] = path_str
        end
        LIBRARY_INITIALIZED[] = true
        return POSEEST_SUCCESS
    catch e
        println(stderr, "Failed to initialize library: $e")
        return POSEEST_ERROR_INVALID_INPUT
    end
end

# Enhanced 6DOF pose estimation
Base.@ccallable function estimate_pose_6dof(
    runway_corners_::Ptr{WorldPointF64},
    projections_::Ptr{ProjectionPointF64},
    num_points::Cint,
    camera_config::CAMERA_CONFIG_C,
    result::Ptr{PoseEstimate_C}
)::Cint
    # try
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


    # Get camera configuration
    camconfig = get_camera_config(camera_config)

    # Perform pose estimation
    sol = estimatepose6dof(runway_corners, projections, camconfig)

    # Convert result back to C struct
    result_c = PoseEstimate_C(
        sol.pos .|> _ustrip(m),
        Rotations.params(sol.rot),
        # Note: sol doesn't seem to have residual_norm and converged fields in the current implementation
        # We'll set defaults for now
        0.0,  # residual_norm
        1     # converged (assume success if no exception)
    )

    # Write result to output pointer
    unsafe_store!(result, result_c)

    return POSEEST_SUCCESS

    # catch e
    #     println(stderr, "Error in estimate_pose_6dof: $e")
    #     if isa(e, BoundsError) || isa(e, ArgumentError)
    #         return POSEEST_ERROR_INVALID_INPUT
    #     else
    #         return POSEEST_ERROR_NO_CONVERGENCE
    #     end
    # end
end

# Enhanced 3DOF pose estimation
Base.@ccallable function estimate_pose_3dof(
    runway_corners_::Ptr{WorldPointF64},
    projections_::Ptr{ProjectionPointF64},
    num_points::Cint,
    known_rotation::Ptr{RotYPRF64},
    camera_config::CAMERA_CONFIG_C,
    result::Ptr{PoseEstimate_C}
)::Cint
    # Validate inputs
    if runway_corners_ == C_NULL || projections_ == C_NULL || known_rotation == C_NULL || result == C_NULL
        return POSEEST_ERROR_INVALID_INPUT
    end

    if num_points < 4
        return POSEEST_ERROR_INSUFFICIENT_POINTS
    end

    # Convert C arrays to Julia arrays
    runway_corners = unsafe_wrap(Array, runway_corners_, num_points) .* 1m
    projections = unsafe_wrap(Array, projections_, num_points) .* 1px
    known_rot_c = unsafe_load(known_rotation)

    # Convert rotation to Julia type
    jl_rotation = RotZYX(known_rot_c[1], known_rot_c[2], known_rot_c[3])

    # Get camera configuration
    camconfig = get_camera_config(camera_config)

    # Perform pose estimation
    sol = estimatepose3dof(runway_corners, projections, jl_rotation, camconfig)

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
end

# Enhanced 6DOF pose estimation with covariance specification
Base.@ccallable function estimate_pose_6dof_with_covariance(
    runway_corners_::Ptr{WorldPointF64},
    projections_::Ptr{ProjectionPointF64},
    num_points::Cint,
    covariance_data::Ptr{Cdouble},
    covariance_type::COVARIANCE_TYPE_C,
    camera_config::CAMERA_CONFIG_C,
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
        noise_model = parse_covariance_data(covariance_data, covariance_type, num_points)

        # Get camera configuration
        camconfig = get_camera_config(camera_config)

        # Perform pose estimation with custom noise model
        sol = estimatepose6dof(runway_corners, projections, camconfig; noise_model=noise_model)

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
        println(stderr, "Error in estimate_pose_6dof_with_covariance: $e")
        if isa(e, BoundsError) || isa(e, ArgumentError)
            return POSEEST_ERROR_INVALID_INPUT
        else
            return POSEEST_ERROR_NO_CONVERGENCE
        end
    end
end

# Enhanced 3DOF pose estimation with covariance specification
Base.@ccallable function estimate_pose_3dof_with_covariance(
    runway_corners_::Ptr{WorldPointF64},
    projections_::Ptr{ProjectionPointF64},
    num_points::Cint,
    known_rotation::Ptr{RotYPRF64},
    covariance_data::Ptr{Cdouble},
    covariance_type::COVARIANCE_TYPE_C,
    camera_config::CAMERA_CONFIG_C,
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
        jl_rotation = RotZYX(known_rot_c[1], known_rot_c[2], known_rot_c[3])

        # Parse covariance specification
        noise_model = parse_covariance_data(covariance_data, covariance_type, num_points)

        # Get camera configuration
        camconfig = get_camera_config(camera_config)

        # Perform pose estimation with custom noise model
        sol = estimatepose3dof(runway_corners, projections, jl_rotation, camconfig; noise_model=noise_model)

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
        println(stderr, "Error in estimate_pose_3dof_with_covariance: $e")
        if isa(e, BoundsError) || isa(e, ArgumentError)
            return POSEEST_ERROR_INVALID_INPUT
        else
            return POSEEST_ERROR_NO_CONVERGENCE
        end
    end
end

# Point projection utility
Base.@ccallable function project_point(
    camera_position::Ptr{WorldPointF64},
    camera_rotation::Ptr{RotYPRF64},
    world_point::Ptr{WorldPointF64},
    camera_config::CAMERA_CONFIG_C,
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

    # Get camera configuration
    camconfig = get_camera_config(camera_config)

    # Project point
    jl_projection = project(jl_cam_pos, jl_cam_rot, jl_world_pt, camconfig)

    # Convert result back to C struct
    result_c = jl_projection .|> _ustrip(px)

    # Write result to output pointer
    unsafe_store!(result, result_c)

    return POSEEST_SUCCESS
end
