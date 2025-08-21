"Camera projection model for runway pose estimation."

using LinearAlgebra
using Unitful, Unitful.DefaultSymbols
using StaticArrays
import Moshi.Match: @match

# Abstract camera configuration interface
abstract type AbstractCameraConfig{S} end

# Camera configuration with type parameter for coordinate system
struct CameraConfig{S} <: AbstractCameraConfig{S}
    focal_length::typeof(1.0mm)
    pixel_size::typeof(1.0μm/px)
    image_width::typeof(1.0px)
    image_height::typeof(1.0px)
end

# Default camera configurations
const CAMERA_CONFIG_CENTERED = CameraConfig{:centered}(25.0u"mm", 3.45u"μm" / 1pixel, 4096pixel, 3000pixel)
const CAMERA_CONFIG_OFFSET = CameraConfig{:offset}(25.0u"mm", 3.45u"μm" / 1pixel, 4096pixel, 3000pixel)

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
        S ∈ (:centered, :offset) || throw(ArgumentError("Coordinate system S must be :centered or :offset, got $S"))
        validate_camera_matrix(matrix) || throw(ArgumentError("Invalid camera matrix"))
        ustrip(width) > 0 || throw(ArgumentError("Image width must be positive"))
        ustrip(height) > 0 || throw(ArgumentError("Image height must be positive"))
        return new{S, T}(matrix, width, height)
    end
end

# Convenience constructor without explicit type parameter
CameraMatrix(S::Symbol, matrix::SMatrix{3, 3, T}, width::WithDims(px), height::WithDims(px)) where {T} = CameraMatrix{S}(matrix, width, height)

# Get optical center based on coordinate system
getopticalcenter(cam::AbstractCameraConfig{:centered}) = SA[0.0px, 0.0px]
getopticalcenter(cam::AbstractCameraConfig{:offset}) = SA[(cam.image_width + 1px) / 2, (cam.image_height + 1px) / 2]
getopticalcenter(cam::CameraMatrix) = SA[cam.matrix[1, 3], cam.matrix[2, 3]]

# Constructor from CameraConfig
function CameraMatrix(config::CameraConfig{S}) where {S}
    f_px = config.focal_length / config.pixel_size |> _uconvert(px)
    cx, cy = getopticalcenter(config)
    sgn = S == :centered ? +1 : -1
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
    return CameraMatrix{S}(matrix, config.image_width, config.image_height)
end

# Constructor from CameraMatrix - inverse conversion
function CameraConfig(camera_matrix::CameraMatrix{S}, pixel_size::typeof(1.0 * u"μm" / 1pixel) = 3.45u"μm" / 1pixel) where {S}
    # Extract focal length in pixels from diagonal elements (ignore off-diagonal terms)
    f_px_x = abs(camera_matrix.matrix[1, 1])  # Take absolute value to handle sign differences
    f_px_y = abs(camera_matrix.matrix[2, 2])
    
    # Use average focal length if x and y are different (assuming square pixels)
    f_px = (f_px_x + f_px_y) / 2
    
    # Convert focal length back to physical units
    focal_length = f_px * pixel_size |> _uconvert(u"mm")
    
    return CameraConfig{S}(
        focal_length,
        pixel_size,
        camera_matrix.image_width,
        camera_matrix.image_height,
    )
end

"Project 3D world point to 2D image coordinates using pinhole camera model."
function project(
        cam_pos::WorldPoint{T}, cam_rot::RotZYX, world_pt::WorldPoint{T′},
        camconfig::CameraConfig{S} = CAMERA_CONFIG_OFFSET
    ) where {T, T′, S}
    cam_pt = world_pt_to_cam_pt(cam_pos, cam_rot, world_pt)
    cam_pt.x <= 0m && throw(BehindCameraException(cam_pt.x))

    # Calculate focal length in pixels
    (; focal_length, pixel_size) = camconfig
    f_pixels = focal_length / pixel_size

    u_centered = f_pixels * (cam_pt.y / cam_pt.x) |> _uconvert(pixel)  # Left positive
    v_centered = f_pixels * (cam_pt.z / cam_pt.x) |> _uconvert(pixel)  # Up positive

    T′′ = typeof(u_centered)
    return @match camconfig begin
        ::CameraConfig{:centered} => ProjectionPoint{T′′, :centered}(u_centered, v_centered)
        ::CameraConfig{:offset} => let
            cx, cy = getopticalcenter(camconfig)
            u, v = -u_centered + cx, -v_centered + cy
            ProjectionPoint{T′′, :offset}(u, v)
        end
    end
end

function project(
        cam_pos::WorldPoint{T}, cam_rot::RotZYX, world_pt::WorldPoint{T′},
        camconfig::CameraMatrix{S, U}
    ) where {T, T′, S, U}
    # Transform to camera coordinates
    cam_pt = world_pt_to_cam_pt(cam_pos, cam_rot, world_pt)
    cam_pt.x <= 0m && throw(BehindCameraException(cam_pt.x))

    # P_norm = [X/Z, Y/Z, 1]^T (unitless) - ustrip needed for Julia's type system
    P_norm = SA[cam_pt.y / cam_pt.x, cam_pt.z / cam_pt.x, 1.0]

    # p_img [px] = K_norm [px] * P_norm [unitless] = [result1, result2, result3] [px]
    image_coords_homogeneous = camconfig.matrix * P_norm

    if abs(image_coords_homogeneous[3]) < 1.0e-12px
        throw(DivideError("Point projects to infinity (homogeneous coordinate near zero)"))
    end

    u, v = image_coords_homogeneous[1:2] / image_coords_homogeneous[3]

    T′′ = typeof(u)
    return ProjectionPoint{T′′, S}(u, v)
end

# Clean dispatch-based coordinate conversion using AbstractCameraConfig
function convertcamconf(to::AbstractCameraConfig{:centered}, from::AbstractCameraConfig{:offset}, proj::ProjectionPoint{T, :offset}) where {T}
    u, v = proj.x, proj.y
    cx, cy = getopticalcenter(from)
    u_centered, v_centered = -(u - cx), -(v - cy)
    return ProjectionPoint{T, :centered}(u_centered, v_centered)
end

function convertcamconf(to::AbstractCameraConfig{:offset}, from::AbstractCameraConfig{:centered}, proj::ProjectionPoint{T, :centered}) where {T}
    u_centered, v_centered = proj.x, proj.y
    cx, cy = getopticalcenter(to)
    u, v = -u_centered + cx, -v_centered + cy
    return ProjectionPoint{T, :offset}(u, v)
end

# Same coordinate system - no conversion needed
convertcamconf(to::AbstractCameraConfig{S}, from::AbstractCameraConfig{S}, proj::ProjectionPoint{T, S}) where {T, S} = proj


"Validate 3x3 matrix for camera projection."
validate_camera_matrix(matrix::SMatrix{3, 3}) = all(!iszero, [matrix[1, 1], matrix[2, 2]]) && (matrix[3, :] ≈ SVector(0, 0, 1)px) && all(isfinite, matrix)
