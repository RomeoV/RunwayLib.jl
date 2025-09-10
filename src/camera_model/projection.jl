"Camera projection model for runway pose estimation."

using LinearAlgebra
using Unitful, Unitful.DefaultSymbols
using StaticArrays
import Moshi.Match: @match

# Abstract camera configuration interface
abstract type AbstractCameraConfig{S} end

# CameraConfig has been removed - use CameraMatrix instead
# Temporary backward compatibility for CAMERA_CONFIG_OFFSET
struct CameraConfig{S} <: AbstractCameraConfig{S}
    focal_length_px::typeof(1.0px)
    image_width::typeof(1.0px)
    image_height::typeof(1.0px)
end

# Only :offset configuration supported
# 25.0mm ÷ 3.45μm = 7246.4 pixels
const CAMERA_CONFIG_OFFSET = CameraConfig{:offset}(7246.4px, 4096px, 3000px)

"Camera model using 3x3 projection matrix with uniform pixel units."
struct CameraMatrix{S, T <: WithDims(px)} <: AbstractCameraConfig{S}
    matrix::SMatrix{3, 3, T, 9}  # 3x3 matrix for normalized coordinate projection
    image_width::typeof(1.0px)
    image_height::typeof(1.0px)

    function CameraMatrix{S}(matrix::SMatrix{3, 3, T}, width::WithDims(px), height::WithDims(px)) where {S, T}
        # Delegate to the validating constructor
        return CameraMatrix{S, T}(matrix, width, height)
    end

    # Inner constructor with validation
    function CameraMatrix{S, T}(matrix::SMatrix{3, 3, T}, width::WithDims(px), height::WithDims(px)) where {S, T}
        Base.isconcretetype(T) || throw(ArgumentError("CameraMatrix eltype must be concrete."))
        S == :offset || throw(ArgumentError("Only :offset coordinate system is supported, got $S"))
        validate_camera_matrix(matrix) || throw(ArgumentError("Invalid camera matrix"))
        ustrip(width) > 0 || throw(ArgumentError("Image width must be positive"))
        ustrip(height) > 0 || throw(ArgumentError("Image height must be positive"))
        return new{S, T}(matrix, width, height)
    end
end

# Convenience constructor without explicit type parameter
CameraMatrix(S::Symbol, matrix::SMatrix{3, 3, T}, width::WithDims(px), height::WithDims(px)) where {T} = CameraMatrix{S}(matrix, width, height)

# Get optical center based on coordinate system
getopticalcenter(cam::AbstractCameraConfig{:offset}) = SA[(cam.image_width + 1px) / 2, (cam.image_height + 1px) / 2]
getopticalcenter(cam::CameraMatrix) = SA[cam.matrix[1, 3], cam.matrix[2, 3]]

# Constructor from CameraConfig - only :offset supported
function CameraMatrix(config::CameraConfig{:offset})
    f_px = config.focal_length_px
    cx, cy = getopticalcenter(config)
    sgn = -1  # Only :offset supported
    # Build uniform camera matrix with [px] units for normalized coordinates
    # K_norm projects unitless normalized coordinates [X/Z, Y/Z, 1] to pixels
    # For centered coordinates: u = f * (Y/X), v = f * (Z/X) - no sign flip needed
    # For offset coordinates:   u = -f * (Y/X) + cx, v = -f * (Z/X) + cy - sign flip needed
    # In offset coordinates, cx and cy will be nonzero.
    matrix = SA[
        (sgn*f_px) 0px        cx
        0px        (sgn*f_px) cy
        0px        0px        1px
    ]
    return CameraMatrix{:offset}(matrix, config.image_width, config.image_height)
end

# Constructor from CameraMatrix - inverse conversion, only :offset supported
CameraConfig(camera_matrix::CameraMatrix{:offset}) = CameraConfig{:offset}(camera_matrix)
function CameraConfig{:offset}(camera_matrix::CameraMatrix{:offset})
    # Extract focal length in pixels from diagonal elements (ignore off-diagonal terms)
    f_px_x = abs(camera_matrix.matrix[1, 1])  # Take absolute value to handle sign differences
    f_px_y = abs(camera_matrix.matrix[2, 2])
    
    # Use average focal length if x and y are different (assuming square pixels)
    focal_length_px = (f_px_x + f_px_y) / 2
    
    return CameraConfig{S′}(
        focal_length_px,
        camera_matrix.image_width,
        camera_matrix.image_height,
    )
end
CameraConfig(cameraconf::CameraConfig{S}) where {S} = cameraconf
CameraConfig{S′}(cameraconf::CameraConfig{S}) where {S, S′} = CameraConfig{S′}(
    cameraconf.focal_length_px,
    cameraconf.image_width,
    cameraconf.image_height
)

"Project 3D world point to 2D image coordinates using pinhole camera model."
function project(
        cam_pos::WorldPoint{T}, cam_rot::RotZYX, world_pt::WorldPoint{T′},
        camconfig::CameraConfig{S} = CAMERA_CONFIG_OFFSET
    ) where {T, T′, S}
    cam_pt = world_pt_to_cam_pt(cam_pos, cam_rot, world_pt)
    cam_pt.x <= 0m && throw(BehindCameraException(cam_pt.x))

    # Get focal length in pixels
    f_pixels = camconfig.focal_length_px

    u_centered = f_pixels * (cam_pt.y / cam_pt.x) |> _uconvert(pixel)  # Left positive
    v_centered = f_pixels * (cam_pt.z / cam_pt.x) |> _uconvert(pixel)  # Up positive

    # Convert to :offset coordinates
    cx, cy = getopticalcenter(camconfig)
    u, v = -u_centered + cx, -v_centered + cy
    T′′ = typeof(u_centered)
    return ProjectionPoint{T′′, :offset}(u, v)
end

function project(
        cam_pos::WorldPoint{T}, cam_rot::RotZYX, world_pt::WorldPoint{T′},
        camconfig::CameraMatrix{S, U}
    ) where {T, T′, S, U}
    # Transform to camera coordinates
    cam_pt = world_pt_to_cam_pt(cam_pos, cam_rot, world_pt)
    cam_pt.x <= 0m && throw(BehindCameraException(cam_pt.x))

    # P_norm = [X/Z, Y/Z, 1]^T (unitless)
    P_norm = SA[cam_pt.y / cam_pt.x, cam_pt.z / cam_pt.x, 1.0]

    # p_img [px] = K_norm [px] * P_norm [unitless] = [result1, result2, result3] [px]
    image_coords = camconfig.matrix * P_norm

    u, v = image_coords[1:2]

    T′′ = typeof(u)
    return ProjectionPoint{T′′, S}(u, v)
end

# Coordinate conversion - only :offset supported
convertcamconf(to::AbstractCameraConfig{:offset}, from::AbstractCameraConfig{:offset}, proj::ProjectionPoint{T, :offset}) where {T} = proj


"Validate 3x3 matrix for camera projection."
validate_camera_matrix(matrix::SMatrix{3, 3}) = all(!iszero, [matrix[1, 1], matrix[2, 2]]) && (matrix[3, :] ≈ SVector(0, 0, 1)px) && all(isfinite, matrix)
