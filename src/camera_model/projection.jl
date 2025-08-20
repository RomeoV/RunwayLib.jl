"Camera projection model for runway pose estimation."

using LinearAlgebra
using Unitful
using StaticArrays
import Moshi.Match: @match

# Abstract camera configuration interface
abstract type AbstractCameraConfig{S} end

# Camera configuration with type parameter for coordinate system
struct CameraConfig{S} <: AbstractCameraConfig{S}
    focal_length::typeof(1.0 * u"mm")
    pixel_size::typeof(1.0 * u"μm" / 1pixel)
    image_width::typeof(1pixel)
    image_height::typeof(1pixel)
    optical_center_u::typeof(1.0pixel)
    optical_center_v::typeof(1.0pixel)
end

# Default camera configurations
const CAMERA_CONFIG_CENTERED = CameraConfig{:centered}(
    25.0 * u"mm",                # Focal length
    3.45 * u"μm" / 1pixel,       # Physical pixel size
    4096 * 1pixel,               # Image width in pixels
    3000 * 1pixel,               # Image height in pixels
    0.0 * 1pixel,                # Principal point x-coordinate (centered)
    0.0 * 1pixel                 # Principal point y-coordinate (centered)
)

const CAMERA_CONFIG_OFFSET = CameraConfig{:offset}(
    25.0 * u"mm",                    # Focal length
    3.45 * u"μm" / 1pixel,          # Physical pixel size
    4096 * 1pixel,               # Image width in pixels
    3000 * 1pixel,               # Image height in pixels
    2047.5 * 1pixel,             # Principal point x-coordinate (image center)
    1499.5 * 1pixel              # Principal point y-coordinate (image center)
)

"Camera model using 3x3 projection matrix with uniform pixel units."
struct CameraMatrix{S,T} <: AbstractCameraConfig{S}
    matrix::SMatrix{3,3,T}  # 3x3 matrix for normalized coordinate projection
    image_width::WithDims(px)
    image_height::WithDims(px)
    
    function CameraMatrix{S}(matrix::SMatrix{3,3,T}, width::WithDims(px), height::WithDims(px)) where {S,T}
        # Delegate to the validating constructor
        CameraMatrix{S,T}(matrix, width, height)
    end
    
    # Inner constructor with validation
    function CameraMatrix{S,T}(matrix::SMatrix{3,3,T}, width::WithDims(px), height::WithDims(px)) where {S,T}
        # Validate matrix dimensions (already enforced by SMatrix type)
        # Validate coordinate system type
        if S ∉ (:centered, :offset)
            throw(ArgumentError("Coordinate system S must be :centered or :offset, got $S"))
        end
        
        # Validate that matrix is reasonable (non-zero focal lengths)
        # Use ustrip to get the magnitude for comparison
        if abs(ustrip(matrix[1,1])) < 1e-6 || abs(ustrip(matrix[2,2])) < 1e-6
            throw(ArgumentError("Camera matrix focal length components (matrix[1,1], matrix[2,2]) must be non-zero"))
        end
        
        # Validate that the bottom row is [0, 0, 1] for proper homogeneous coordinates
        # Note: matrix[3,3] should have units px and magnitude 1.0
        if abs(ustrip(matrix[3,1])) > 1e-6 || abs(ustrip(matrix[3,2])) > 1e-6 || abs(ustrip(matrix[3,3]) - 1.0) > 1e-6
            throw(ArgumentError("Camera matrix bottom row should be [0, 0, 1], got [$(matrix[3,1]), $(matrix[3,2]), $(matrix[3,3])]"))
        end
        
        # Validate image dimensions
        if ustrip(width) <= 0.0 || ustrip(height) <= 0.0
            throw(ArgumentError("Image dimensions must be positive"))
        end
        
        new{S,T}(matrix, width, height)
    end
end

# Convenience constructor without explicit type parameter
CameraMatrix(S::Symbol, matrix::SMatrix{3,3,T}, width::WithDims(px), height::WithDims(px)) where {T} = CameraMatrix{S}(matrix, width, height)

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
        # points left and up
        ::CameraConfig{:centered} => let
            ProjectionPoint{T′′, :centered}(u_centered, v_centered)
        end
        # points right and down
        ::CameraConfig{:offset} => let
            u = -u_centered + camconfig.optical_center_u
            v = -v_centered + camconfig.optical_center_v
            ProjectionPoint{T′′, :offset}(u, v)
        end

    end
end

function project(
        cam_pos::WorldPoint{T}, cam_rot::RotZYX, world_pt::WorldPoint{T′},
        camconfig::CameraMatrix{S,U}
    ) where {T, T′, S, U}
    # Transform to camera coordinates
    cam_pt = world_pt_to_cam_pt(cam_pos, cam_rot, world_pt)
    cam_pt.x <= 0m && throw(BehindCameraException(cam_pt.x))
    
    # Step 1: Normalize by Z to get unitless normalized coordinates
    # P_norm = [X/Z, Y/Z, 1]^T (unitless) - ustrip needed for Julia's type system
    P_norm = SA[ustrip(cam_pt.y/cam_pt.x), ustrip(cam_pt.z/cam_pt.x), 1.0]
    
    # Step 2: Apply uniform [px] matrix to unitless vector → [px] result
    # p_img [px] = K_norm [px] * P_norm [unitless] = [result1, result2, result3] [px]
    image_coords_homogeneous = camconfig.matrix * P_norm
    
    # Step 3: Check homogeneous coordinate (compare px with dimensionless threshold)
    # ustrip needed because we're comparing a quantity with units to a dimensionless threshold
    if abs(ustrip(image_coords_homogeneous[3])) < 1e-12
        throw(DivideError("Point projects to infinity (homogeneous coordinate near zero)"))
    end
    
    # Step 4: Normalize to get dimensionless pixel coordinates
    # Try natural division first: px/px should → dimensionless
    u = image_coords_homogeneous[1] / image_coords_homogeneous[3]
    v = image_coords_homogeneous[2] / image_coords_homogeneous[3]
    
    # Convert to appropriate type
    T′′ = typeof(u)
    
    # The matrix should be constructed appropriately for the target coordinate system
    return ProjectionPoint{T′′, S}(u, v)
end

# Clean dispatch-based coordinate conversion using AbstractCameraConfig
function convertcamconf(to::AbstractCameraConfig{:centered}, from::AbstractCameraConfig{:offset}, proj::ProjectionPoint{T, :offset}) where {T}
    u, v = proj.x, proj.y
    # For CameraMatrix, extract principal point from matrix; for CameraConfig, use fields
    cx = from isa CameraMatrix ? from.matrix[1,3] : from.optical_center_u
    cy = from isa CameraMatrix ? from.matrix[2,3] : from.optical_center_v
    u_centered = -(u - cx)
    v_centered = -(v - cy)
    return ProjectionPoint{T, :centered}(u_centered, v_centered)
end

function convertcamconf(to::AbstractCameraConfig{:offset}, from::AbstractCameraConfig{:centered}, proj::ProjectionPoint{T, :centered}) where {T}
    u_centered, v_centered = proj.x, proj.y
    # For CameraConfig, use optical_center fields; for CameraMatrix, need to get from somewhere else
    cx = to isa CameraConfig ? to.optical_center_u : throw(ArgumentError("Cannot convert to CameraMatrix{:offset} without knowing target principal point"))
    cy = to isa CameraConfig ? to.optical_center_v : throw(ArgumentError("Cannot convert to CameraMatrix{:offset} without knowing target principal point"))
    u = -u_centered + cx
    v = -v_centered + cy
    return ProjectionPoint{T, :offset}(u, v)
end

# Same coordinate system - no conversion needed
convertcamconf(to::AbstractCameraConfig{S}, from::AbstractCameraConfig{S}, proj::ProjectionPoint{T, S}) where {T, S} = proj

"Convert parametric CameraConfig to equivalent CameraMatrix."
function camera_config_to_matrix(config::CameraConfig{S}) where S
    # Calculate focal length in pixels (for normalized coordinates)
    f_pixels_raw = config.focal_length / config.pixel_size
    # Convert to pixels by using the pixel unit from this module
    f_pixels_quantity = f_pixels_raw |> _uconvert(pixel) 
    f_pixels_value = ustrip(f_pixels_quantity)  # Extract numerical value
    
    # Extract principal point coordinates in pixels (already in correct units)
    cx_value = ustrip(config.optical_center_u)  
    cy_value = ustrip(config.optical_center_v)
    
    # Build uniform camera matrix with [px] units for normalized coordinates
    # K_norm projects unitless normalized coordinates [X/Z, Y/Z, 1] to pixels
    # [fx  s   cx] [px px px]
    # [0   fy  cy] [px px px] 
    # [0   0   1 ] [px px px]
    # Build matrix based on coordinate system
    if S == :centered
        # For centered coordinates: u = f * (Y/X), v = f * (Z/X) - no sign flip needed
        matrix = @SMatrix [
            f_pixels_value*px  0.0px              cx_value*px;
            0.0px              f_pixels_value*px  cy_value*px;
            0.0px              0.0px              1.0px
        ]
    else  # :offset
        # For offset coordinates: u = -f * (Y/X) + cx, v = -f * (Z/X) + cy - sign flip needed
        matrix = @SMatrix [
            -f_pixels_value*px  0.0px               cx_value*px;
            0.0px               -f_pixels_value*px  cy_value*px;
            0.0px               0.0px               1.0px
        ]
    end
    
    return CameraMatrix{S}(matrix, config.image_width, config.image_height)
end

"Validate 3x3 matrix for camera projection."
function validate_camera_matrix(matrix::SMatrix{3,3,Float64})
    # Check for NaN or Inf values
    if any(!isfinite, matrix)
        return false
    end
    
    # Check focal length components
    if abs(matrix[1,1]) < 1e-6 || abs(matrix[2,2]) < 1e-6
        return false
    end
    
    # Check bottom row format [0, 0, 1]
    if abs(matrix[3,1]) > 1e-6 || abs(matrix[3,2]) > 1e-6 || abs(matrix[3,3] - 1.0) > 1e-6
        return false
    end
    
    return true
end
