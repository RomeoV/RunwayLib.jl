# Enhanced C API for RunwayLib pose estimation
# Provides C-callable functions for external language bindings

using StaticArrays
using Rotations
using Unitful, Unitful.DefaultSymbols
using CEnum

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

function get_camera_config(config_type::CAMERA_CONFIG_C)
    if config_type == CAMERA_CONFIG_CENTERED_C
        return CAMERA_CONFIG_CENTERED
    elseif config_type == CAMERA_CONFIG_OFFSET_C
        return CAMERA_CONFIG_OFFSET
    else
        throw(ArgumentError("Invalid camera config type: $config_type"))
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
