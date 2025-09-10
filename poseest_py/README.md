# PoseEst: Python Wrapper for Runway Pose Estimation

PoseEst is a Python package that provides a clean, pythonic interface to a high-performance runway pose estimation library implemented in Julia. The package allows you to estimate 6-DOF and 3-DOF aircraft poses from runway corner projections in camera images.

## Features

- **High Performance**: Built on a Julia backend for optimal computational performance
- **Relocatable**: Self-contained with no external Julia installation required  
- **Simple API**: Clean Python interface that hides the complexity of the underlying C/Julia code
- **Camera Matrix Support**: Flexible camera matrix specification with offset coordinate system
- **Robust Error Handling**: Comprehensive error checking and informative error messages

## Installation

### From PyPI (when available)
```bash
pip install poseest
```

### From Source
```bash
# Clone the repository (or extract from archive)
cd poseest-py/

# Install in development mode
pip install -e .

# Or install normally
pip install .
```

## Quick Start

```python
import poseest

# Define runway corners in world coordinates (meters)
runway_corners = [
    poseest.WorldPoint(0.0, -50.0, 0.0),      # Near left
    poseest.WorldPoint(0.0, 50.0, 0.0),       # Near right  
    poseest.WorldPoint(3000.0, 50.0, 0.0),    # Far right
    poseest.WorldPoint(3000.0, -50.0, 0.0),   # Far left
]

# Image projections of these corners (pixels)
projections = [
    poseest.ProjectionPoint(245.3, 180.7),
    poseest.ProjectionPoint(251.2, 185.1),
    poseest.ProjectionPoint(402.1, 201.4),
    poseest.ProjectionPoint(395.8, 196.9),
]

# Create camera matrix for offset coordinate system
camera_matrix = poseest.CameraMatrix(
    matrix=[
        [7246.4, 0.0, 2048.5],  # fx, 0, cx
        [0.0, 7246.4, 1500.5], # 0, fy, cy  
        [0.0, 0.0, 1.0]         # 0, 0, 1
    ],
    image_width=4096.0,
    image_height=3000.0,
    coordinate_system='offset'
)

# Estimate 6DOF pose (position + orientation)
pose = poseest.estimate_pose_6dof(
    runway_corners, 
    projections, 
    camera_matrix
)

print(f"Aircraft Position: ({pose.position.x:.1f}, {pose.position.y:.1f}, {pose.position.z:.1f}) meters")
print(f"Aircraft Attitude: yaw={pose.rotation.yaw:.3f}, pitch={pose.rotation.pitch:.3f}, roll={pose.rotation.roll:.3f} radians")
```

## API Reference

### Data Types

#### WorldPoint
Represents a 3D point in runway-relative world coordinates.
```python
point = poseest.WorldPoint(x, y, z)  # All in meters
```

#### ProjectionPoint  
Represents a 2D point in image coordinates.
```python
projection = poseest.ProjectionPoint(u, v)  # All in pixels
```

#### Rotation
Represents aircraft attitude as ZYX Euler angles.
```python
rotation = poseest.Rotation(yaw, pitch, roll)  # All in radians
```

#### PoseEstimate
Contains the complete pose estimation result.
```python
pose.position     # WorldPoint: aircraft position
pose.rotation     # Rotation: aircraft attitude  
pose.residual     # float: optimization residual
pose.converged    # bool: whether optimization converged
```

### Functions

#### estimate_pose_6dof()
Estimates both position and orientation from runway corner projections.
```python
pose = poseest.estimate_pose_6dof(
    runway_corners,    # List[WorldPoint]: 4+ runway corners in world coords
    projections,       # List[ProjectionPoint]: corresponding image projections
    camera_matrix      # CameraMatrix: camera matrix configuration
)
```

#### estimate_pose_3dof() 
Estimates only position when orientation is known.
```python
pose = poseest.estimate_pose_3dof(
    runway_corners,    # List[WorldPoint]: 3+ runway corners
    projections,       # List[ProjectionPoint]: corresponding projections  
    known_rotation,    # Rotation: known aircraft attitude
    camera_matrix      # CameraMatrix: camera matrix configuration
)
```

#### project_point()
Projects a 3D world point to 2D image coordinates.
```python
projection = poseest.project_point(
    camera_position,   # WorldPoint: camera position in world coords
    camera_rotation,   # Rotation: camera attitude  
    world_point,       # WorldPoint: 3D point to project
    camera_matrix      # CameraMatrix: camera matrix configuration
)
```

### Camera Matrix Configuration

The package uses camera matrix specifications with the offset coordinate system:

- **CameraMatrix**: 3x3 camera matrix with intrinsic parameters
- **Coordinate System**: Only 'offset' is supported (origin at top-left corner, Y-axis pointing down)

```python
camera_matrix = poseest.CameraMatrix(
    matrix=[
        [fx, 0.0, cx],  # Focal length x, principal point x
        [0.0, fy, cy],  # Focal length y, principal point y  
        [0.0, 0.0, 1.0] # Homogeneous coordinate
    ],
    image_width=4096.0,
    image_height=3000.0,
    coordinate_system='offset'
)
```

## Error Handling

The package provides detailed error messages for common issues:

```python
try:
    pose = poseest.estimate_pose_6dof(corners, projections, config)
except poseest.PoseEstimationError as e:
    print(f"Pose estimation failed: {e}")
except poseest.InvalidInputError as e:
    print(f"Invalid input: {e}")
```

## Performance Notes

- First call to any function includes Julia initialization (~50-100ms)
- Subsequent calls are much faster (~1-10ms depending on problem size)
- The library is thread-safe for multiple simultaneous calls
- Memory usage is minimal as the Julia runtime is embedded

## Algorithm Details

The pose estimation uses a robust nonlinear optimization approach:

1. **6DOF Estimation**: Simultaneous optimization over position and attitude using Levenberg-Marquardt
2. **3DOF Estimation**: Position-only optimization when attitude is constrained
3. **Outlier Rejection**: Robust cost function to handle measurement noise
4. **Multiple Initializations**: Automatic restart from different initial guesses

The underlying Julia implementation leverages:
- Automatic differentiation for exact gradients
- Sparse linear algebra for efficiency  
- Modern optimization algorithms
- Comprehensive unit and integration tests

## Requirements

- Python 3.8+
- NumPy >= 1.20.0
- No Julia installation required (self-contained)

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## Citation

If you use this package in academic work, please cite:

```bibtex
@software{poseest_python,
  title={PoseEst: Python Wrapper for Runway Pose Estimation},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/poseest-py}
}
```