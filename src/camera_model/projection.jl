"""
Camera projection model for runway pose estimation.

This module implements the pinhole camera model for projecting 3D points
to 2D image coordinates, with support for units and realistic camera parameters.
"""

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

"""
    CameraMatrix{S} <: AbstractCameraConfig{S}

Camera model using a 3x4 projection matrix for direct matrix-based projection.

# Fields
- `matrix::SMatrix{3,4,Float64}`: 3x4 camera projection matrix
- `image_width::typeof(1pixel)`: Image width for coordinate system handling
- `image_height::typeof(1pixel)`: Image height for coordinate system handling

# Type Parameter
- `S`: Coordinate system type (`:centered` or `:offset`)

# Matrix Format
The 3x4 projection matrix should be in the standard computer vision format:
```
[fx  s   cx  tx]
[0   fy  cy  ty]  
[0   0   1   tz]
```
where:
- fx, fy: focal lengths in pixels
- cx, cy: principal point coordinates
- s: skew parameter (usually 0)
- tx, ty, tz: translation components (usually 0 for intrinsic matrix)

# Coordinate Systems
For `:centered` coordinates, the principal point should be relative to image center.
For `:offset` coordinates, the principal point should be relative to top-left corner.

# Examples
```julia
# Create camera matrix for offset coordinates
matrix = @SMatrix [
    2000.0  0.0     2048.0  0.0;
    0.0     2000.0  1500.0  0.0;
    0.0     0.0     1.0     0.0
]
cam = CameraMatrix{:offset}(matrix, 4096px, 3000px)

# Use in projection
result = project(camera_pos, camera_rot, world_point, cam)
```
"""
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

"""
    project(cam_pos::WorldPoint, cam_rot::RotZYX, world_pt::WorldPoint, 
            config::CameraConfig{S}=CAMERA_CONFIG_OFFSET) -> ProjectionPoint{T, S}

Project a 3D world point to 2D image coordinates using pinhole camera model.

# Arguments
- `cam_pos::WorldPoint`: Camera position in world coordinates
- `cam_rot::RotZYX`: Camera orientation (ZYX Euler angles)
- `world_pt::WorldPoint`: 3D point to project in world coordinates
- `config::CameraConfig{S}`: Camera configuration with coordinate system type

# Returns
- `ProjectionPoint{T, S}`: 2D image coordinates in pixels with matching coordinate system

# Coordinate Systems
For `:centered` coordinates:
- Origin at image center
- X-axis: Left (positive follows cross-track left convention)
- Y-axis: Up (positive follows height up convention)

For `:offset` coordinates:
- Origin at top-left corner
- X-axis: Right (positive to the right)
- Y-axis: Down (positive downward)

# Algorithm
1. Transform world point to camera coordinates
2. Apply pinhole projection model
3. Convert to appropriate coordinate system

# Exceptions
- `DivideError`: If point is at or behind the camera (X ≤ 0)
- `DomainError`: If projection results in invalid coordinates

# Examples
```julia
# Project to centered coordinates
cam_pos = WorldPoint(-500.0u"m", 0.0u"m", 100.0u"m")
cam_rot = RotZYX(0.0, 0.1, 0.0)
runway_corner = WorldPoint(0.0u"m", 25.0u"m", 0.0u"m")

centered_coords = project(cam_pos, cam_rot, runway_corner, CAMERA_CONFIG_CENTERED)
offset_coords = project(cam_pos, cam_rot, runway_corner, CAMERA_CONFIG_OFFSET)
```
"""
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

"""
    project(cam_pos::WorldPoint, cam_rot::RotZYX, world_pt::WorldPoint, 
            camconfig::CameraMatrix{S}) -> ProjectionPoint{T, S}

Project a 3D world point to 2D image coordinates using a camera projection matrix.

# Arguments
- `cam_pos::WorldPoint`: Camera position in world coordinates
- `cam_rot::RotZYX`: Camera orientation (ZYX Euler angles)
- `world_pt::WorldPoint`: 3D point to project in world coordinates
- `camconfig::CameraMatrix{S}`: Camera matrix configuration

# Returns
- `ProjectionPoint{T, S}`: 2D image coordinates in pixels with matching coordinate system

# Algorithm
1. Transform world point to camera coordinates
2. Convert to homogeneous coordinates
3. Apply camera matrix multiplication
4. Normalize by homogeneous coordinate
5. Handle coordinate system conversion if needed

# Exceptions
- `BehindCameraException`: If point is at or behind the camera
- `DivideError`: If homogeneous coordinate is zero
"""
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

function convertcamconf(to::CameraConfig{:centered}, from::CameraConfig{:offset}, proj::ProjectionPoint{T, :offset}) where {T}
    u, v = proj.x, proj.y
    u_centered = -(u - from.optical_center_u)
    v_centered = -(v - from.optical_center_v)
    return ProjectionPoint{T, :centered}(u_centered, v_centered)
end

function convertcamconf(to::CameraConfig{:offset}, from::CameraConfig{:centered}, proj::ProjectionPoint{T, :centered}) where {T}
    u_centered, v_centered = proj.x, proj.y
    u = -u_centered + to.optical_center_u
    v = -v_centered + to.optical_center_v
    return ProjectionPoint{T, :offset}(u, v)
end
convertcamconf(to::CameraConfig{S}, from::CameraConfig{S}, proj::ProjectionPoint{T, S}) where {T, S} = proj

# Add support for CameraMatrix conversion - delegate to coordinate system conversion
function convertcamconf(to::CameraConfig{S1}, from::CameraMatrix{S2}, proj::ProjectionPoint{T, S2}) where {T, S1, S2}
    # For CameraMatrix, we only need to handle coordinate system conversion, not intrinsic parameter conversion
    # since CameraMatrix doesn't have the same parametric structure as CameraConfig
    if S1 == S2
        return proj  # Same coordinate system, no conversion needed
    elseif S1 == :centered && S2 == :offset
        # Convert from offset to centered: u_centered = -(u - cx), v_centered = -(v - cy)
        # We need to extract the principal point from the CameraMatrix
        # For a CameraMatrix, the principal point is in the matrix elements [1,3] and [2,3]
        cx = from.matrix[1,3]
        cy = from.matrix[2,3]
        u_centered = -(proj.x - cx)
        v_centered = -(proj.y - cy)
        return ProjectionPoint{T, :centered}(u_centered, v_centered)
    elseif S1 == :offset && S2 == :centered
        # Convert from centered to offset: u = -u_centered + cx, v = -v_centered + cy
        # We need the target camera config's principal point - but we only have the destination config
        u = -proj.x + to.optical_center_u
        v = -proj.y + to.optical_center_v
        return ProjectionPoint{T, :offset}(u, v)
    else
        error("Unsupported coordinate system conversion from $S2 to $S1")
    end
end

# Support for converting from CameraConfig to CameraMatrix (delegate to CameraConfig methods)
function convertcamconf(to::CameraMatrix{S1}, from::CameraConfig{S2}, proj::ProjectionPoint{T, S2}) where {T, S1, S2}
    if S1 == S2
        return proj  # Same coordinate system, no conversion needed
    else
        # Convert to an equivalent CameraConfig first, then handle coordinate conversion
        temp_config = CameraConfig{S1}(
            25.0u"mm", 3.45u"μm"/pixel, from.image_width, from.image_height,
            S1 == :centered ? 0.0pixel : from.image_width/2, 
            S1 == :centered ? 0.0pixel : from.image_height/2
        )
        return convertcamconf(temp_config, from, proj)
    end
end

# Support for CameraMatrix to CameraMatrix (mostly just coordinate system conversion)
convertcamconf(to::CameraMatrix{S}, from::CameraMatrix{S}, proj::ProjectionPoint{T, S}) where {T, S} = proj

"""
    camera_config_to_matrix(config::CameraConfig{S}) -> CameraMatrix{S}

Convert a parametric CameraConfig to an equivalent CameraMatrix.

# Arguments
- `config::CameraConfig{S}`: Parametric camera configuration

# Returns
- `CameraMatrix{S}`: Equivalent camera matrix configuration

# Algorithm
Constructs a 3x4 camera intrinsic matrix from the focal length, pixel size, 
and principal point parameters in the CameraConfig.

# Examples
```julia
# Convert the default offset configuration
matrix_config = camera_config_to_matrix(CAMERA_CONFIG_OFFSET)

# Both should produce identical results
result1 = project(cam_pos, cam_rot, world_pt, CAMERA_CONFIG_OFFSET)
result2 = project(cam_pos, cam_rot, world_pt, matrix_config)
```
"""
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

"""
    validate_camera_matrix(matrix::SMatrix{3,3,Float64}) -> Bool

Validate that a 3x3 matrix is suitable for camera projection.

# Arguments
- `matrix::SMatrix{3,3,Float64}`: Camera projection matrix to validate

# Returns
- `Bool`: true if matrix is valid, false otherwise

# Validation Checks
- Bottom row should be [0, 0, 1] for proper homogeneous coordinates
- Focal length components (matrix[1,1], matrix[2,2]) should be non-zero
- Matrix should not contain NaN or Inf values

# Examples
```julia
good_matrix = @SMatrix [2000.0 0.0 1024.0; 0.0 2000.0 512.0; 0.0 0.0 1.0]
bad_matrix = @SMatrix [0.0 0.0 1024.0; 0.0 2000.0 512.0; 0.0 0.0 1.0]

validate_camera_matrix(good_matrix)  # true
validate_camera_matrix(bad_matrix)   # false (zero focal length)
```
"""
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
