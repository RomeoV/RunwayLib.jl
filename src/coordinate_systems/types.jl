"""
Coordinate system type definitions for runway pose estimation.

This module defines the three main coordinate systems used:
- WorldPoint: Runway-relative world coordinates
- CameraPoint: Camera-centric coordinates  
- ProjectionPoint: Image projection coordinates
"""


"""
    WorldPoint{T} <: FieldVector{3, T}

Point in world coordinate system (runway-relative).

# Fields
- `x::T`: Along-track distance (positive towards far end of runway)
- `y::T`: Cross-track distance (positive towards right side of runway)
- `z::T`: Height above runway surface (positive upward)

# Units
Typically uses meters (u"m") for all coordinates.

# Examples
```julia
# Create a point 500m before runway threshold, 10m right of centerline, 100m high
wp = WorldPoint(-500.0u"m", 10.0u"m", 100.0u"m")

# Access coordinates
println("Along-track: ", wp.x)
println("Cross-track: ", wp.y) 
println("Height: ", wp.z)

# Arithmetic operations
wp2 = WorldPoint(100.0u"m", 0.0u"m", 50.0u"m")
wp_sum = wp + wp2  # Element-wise addition
wp_scaled = 2.0 * wp  # Scalar multiplication
```
"""
struct WorldPoint{T} <: FieldVector{3,T}
    x::T  # Along-track distance
    y::T  # Cross-track distance
    z::T  # Height above runway
end

"""
    CameraPoint{T} <: FieldVector{3, T}

Point in camera coordinate system.

# Fields
- `x::T`: Camera forward direction (positive towards scene)
- `y::T`: Camera right direction (positive to the right)
- `z::T`: Camera down direction (positive downward)

# Units
Typically uses meters (u"m") for all coordinates.

# Coordinate System Convention
- X-axis: Forward (into the scene)
- Y-axis: Right (to the right of the camera)
- Z-axis: Down (downward from camera)

This follows the standard computer vision convention.

# Examples
```julia
# Point 10m in front of camera, 2m to the right, 1m below
cp = CameraPoint(10.0u"m", 2.0u"m", 1.0u"m")

# Access coordinates
println("Forward: ", cp.x)
println("Right: ", cp.y)
println("Down: ", cp.z)
```
"""
struct CameraPoint{T} <: FieldVector{3,T}
    x::T  # Camera forward direction
    y::T  # Camera right direction
    z::T  # Camera down direction
end

"""
    ProjectionPoint{T, S} <: FieldVector{2, T}

Point in image projection coordinate system.

# Type Parameters
- `T`: Numeric type for coordinates
- `S`: Coordinate system type (only `:offset` supported)

# Fields
- `x::T`: Image x-coordinate (horizontal pixel position)
- `y::T`: Image y-coordinate (vertical pixel position)

# Units
Typically uses pixels (1pixel) for coordinates.

# Coordinate System Convention
For `:offset` coordinates:
- Origin at top-left corner of image
- X-axis: Horizontal (positive to the right)
- Y-axis: Vertical (positive downward)

# Examples
```julia
# Offset coordinates (origin at top-left)
pp_offset = ProjectionPoint{Float64, :offset}(1024.0*1pixel, 768.0*1pixel)

# Access coordinates
println("X: ", pp_offset.x)
println("Y: ", pp_offset.y)
```
"""
struct ProjectionPoint{T,S} <: FieldVector{2,T}
    x::T  # Image x-coordinate (pixels)
    y::T  # Image y-coordinate (pixels)
end

# Convenience constructors for common use cases
# WorldPoint(x, y, z) = WorldPoint{typeof(x)}(x, y, z)
# WorldPoint(xs::AbstractVector{T}) = WorldPoint{T}(xs)
# CameraPoint(x, y, z) = CameraPoint{typeof(x)}(x, y, z)
# CameraPoint(xs::AbstractVector{T}) = CameraPoint{T}(xs)
# ProjectionPoint(x, y) = ProjectionPoint{typeof(x), :offset}(x, y)  # Default to offset coordinates
ProjectionPoint(xy::AbstractVector{T}) where {T} = ProjectionPoint{T,:offset}(xy)
ProjectionPoint(type::Symbol, x::T, y::T) where {T} = ProjectionPoint{T,type}(x, y)  # Default to offset coordinates
ProjectionPoint{T}(x, y) where {T} = ProjectionPoint{T,:offset}(x, y)

# Base.BroadcastStyle(::Type{<:WorldPoint}) = Broadcast.ArrayStyle{WorldPoint}()
# Base.similar(::Broadcast.Broadcasted{Broadcast.ArrayStyle{WorldPoint}}, ::Type{T}) where {T} = WorldPoint{T}(undef)
# Base.similar(::Broadcast.Broadcasted{Broadcast.ArrayStyle{WorldPoint}}, ::Type{T}) where {T} =
#     WorldPoint{T}(undef, undef, undef)


similar_type(::Type{<:ProjectionPoint{T, :offset}}, ::Type{T′}, s::Size{S}) where {T, T′, S} = ProjectionPoint{T′, :offset}
similar_type(::Type{<:WorldPoint}, ::Type{T}, s::Size{S}) where {T, S} = WorldPoint{T}
similar_type(::Type{<:CameraPoint}, ::Type{T}, s::Size{S}) where {T, S} = CameraPoint{T}


struct Line{RT<:WithDims(px),TT<:WithDims(rad)}
    r::RT
    theta::TT
end
