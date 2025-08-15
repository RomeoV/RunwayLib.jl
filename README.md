# RunwayPoseEstimation.jl

A Julia library for aircraft pose estimation using runway corner detection from camera images, with integrated uncertainty quantification and integrity monitoring.

## Overview

This library provides a complete pipeline for estimating aircraft pose (position and orientation) relative to runways using computer vision techniques. The system processes camera images from aircraft, detects runway corners using machine learning models, and estimates 6-DOF pose with uncertainty quantification using the RAIM (Receiver Autonomous Integrity Monitoring) framework.

## Architecture

The library is organized into several key modules:

### 1. Data Management (`RunwayData`)
- **Purpose**: Handle runway reference data and flight test datasets
- **Components**:
  - `RunwayDatabase`: Load and manage runway specifications from Excel files
  - `FlightData`: Process CSV files containing flight test data with predictions
  - Data validation and preprocessing utilities

### 2. Coordinate Systems (`CoordinateSystems`)
- **Purpose**: Define and transform between different coordinate frames
- **Components**:
  - `WorldPoint`: World coordinate system (runway-relative)
  - `CameraPoint`: Camera-centric coordinate system  
  - `ProjectionPoint`: Image projection coordinate system
  - Transformation functions between coordinate systems

### 3. Camera Model (`CameraModel`)
- **Purpose**: Handle camera projection and calibration
- **Components**:
  - Camera intrinsic parameters (focal length, pixel size, optical center)
  - Projection functions from 3D world points to 2D image coordinates
  - Inverse projection utilities

### 4. Pose Estimation (`PoseEstimation`)
- **Purpose**: Estimate aircraft pose from runway corner observations
- **Components**:
  - 6-DOF pose estimation (position + orientation)
  - 3-DOF position-only estimation (with known orientation)
  - Nonlinear least squares optimization
  - Initial guess generation and convergence handling

### 5. Uncertainty Quantification (`UncertaintyQuantification`)
- **Purpose**: Propagate pixel uncertainties to pose uncertainties
- **Components**:
  - Integration with `ProbabilisticParameterEstimators.jl`
  - Noise model construction from pixel covariances
  - Bayesian and frequentist uncertainty estimation methods
  - Calibration assessment tools

### 6. Integrity Monitoring (`IntegrityMonitoring`)
- **Purpose**: Implement RAIM-based integrity checks
- **Components**:
  - Residual computation and chi-squared test statistics
  - Fault detection and exclusion algorithms
  - Integrity risk assessment
  - Performance metrics (false alarm rate, missed detection rate)

### 7. Visualization (`Visualization`)
- **Purpose**: Plotting and analysis tools
- **Components**:
  - Runway corner visualization with uncertainty ellipses
  - Pose trajectory plotting
  - Error distribution analysis
  - Calibration plots and diagnostic visualizations

## Key Data Structures

### Coordinate System Types
```julia
struct WorldPoint{T <: Real} <: FieldVector{3, T}
    x::T  # Along-track distance
    y::T  # Cross-track distance  
    z::T  # Height above runway
end

struct CameraPoint{T <: Real} <: FieldVector{3, T}
    x::T  # Camera forward direction
    y::T  # Camera right direction
    z::T  # Camera down direction
end

struct ProjectionPoint{T <: Real} <: FieldVector{2, T}
    x::T  # Image x-coordinate (pixels)
    y::T  # Image y-coordinate (pixels)
end
```

### Runway Specification
```julia
struct RunwaySpec
    icao_code::String
    length_m::Float64
    width_m::Float64
    threshold_elevation_m::Float64
    true_bearing_deg::Float64
end
```

### Pose Estimate
```julia
struct PoseEstimate
    position::WorldPoint{Float64}
    orientation::RotZYX{Float64}
    uncertainty::MvNormal  # Joint position-orientation uncertainty
    residual_norm::Float64
    converged::Bool
end
```

## Workflow

### 1. Data Loading and Preprocessing
```julia
# Load runway database
runway_db = load_runway_database("runway_data.xlsx")

# Load flight test data
flight_data = load_flight_data("flight_test_results.csv")

# Preprocess and validate data
validated_data = preprocess_flight_data(flight_data, runway_db)
```

### 2. Pose Estimation
```julia
# Extract runway corners from ML predictions
corners_image = extract_runway_corners(flight_data_row)

# Get runway specification
runway_spec = get_runway_spec(runway_db, flight_data_row.airport_runway)

# Estimate pose
pose_estimate = estimate_pose_6dof(
    runway_spec, 
    corners_image, 
    pixel_uncertainties;
    initial_guess=initial_pose_guess
)
```

### 3. Uncertainty Quantification
```julia
# Create noise model from pixel uncertainties
noise_model = create_noise_model(pixel_covariances)

# Estimate pose with uncertainty
pose_distribution = estimate_pose_with_uncertainty(
    runway_spec,
    corners_image, 
    noise_model,
    prior_distribution
)
```

### 4. Integrity Monitoring
```julia
# Compute RAIM statistic
raim_statistic = compute_raim_statistic(
    pose_estimate,
    runway_spec,
    corners_image,
    noise_model
)

# Check integrity
integrity_flag = check_integrity(raim_statistic, significance_level=0.05)
```

### 5. Analysis and Visualization
```julia
# Visualize results
plot_runway_corners_with_uncertainty(corners_image, pixel_uncertainties)
plot_pose_trajectory(pose_estimates)
plot_integrity_statistics(raim_statistics, confusion_matrix_labels)

# Assess calibration
calibration_results = assess_calibration(pose_distributions, ground_truth_poses)
```

## Dependencies

### Core Dependencies
- `LinearAlgebra`: Matrix operations and decompositions
- `Distributions`: Probability distributions and sampling
- `Rotations`: 3D rotation representations and operations
- `StaticArrays`: Efficient small vector/matrix operations
- `NonlinearSolve`: Nonlinear optimization for pose estimation

### Specialized Dependencies  
- `ProbabilisticParameterEstimators`: Uncertainty quantification (included in lib/)
- `ForwardDiff`: Automatic differentiation for Jacobian computation
- `DifferentiationInterface`: Generic differentiation interface

### Data and Visualization
- `CSV`, `DataFrames`: Data loading and manipulation
- `XLSX`: Excel file reading for runway database
- `CairoMakie`: High-quality plotting and visualization
- `AlgebraOfGraphics`: Grammar of graphics for complex plots

## Configuration

### Camera Parameters
```julia
const CAMERA_CONFIG = (
    focal_length_m = 25e-3,           # 25mm focal length
    pixel_size_m = 0.00345e-3,        # Pixel physical size
    image_width_px = 4096,            # Image width in pixels
    image_height_px = 3000,           # Image height in pixels
    optical_center_u_px = 2047.5,     # Principal point x
    optical_center_v_px = 1499.5      # Principal point y
)
```

### Estimation Parameters
```julia
const ESTIMATION_CONFIG = (
    max_iterations = 100,
    convergence_tolerance = 1e-6,
    initial_guess_noise_std = [100.0, 10.0, 5.0, 0.1, 0.1, 0.1], # pos + rot
    integrity_significance_level = 0.05
)
```

## Testing Strategy

### Unit Tests
- Coordinate system transformations
- Camera projection accuracy
- Pose estimation convergence
- Uncertainty propagation correctness

### Integration Tests  
- End-to-end pipeline with synthetic data
- Comparison with ground truth on real flight data
- Calibration assessment on validation datasets

### Performance Tests
- Computational efficiency benchmarks
- Memory usage profiling
- Scalability with dataset size

## Usage Examples

### Basic Pose Estimation
```julia
using RunwayPoseEstimation

# Load data
runway_db = load_runway_database("data/runway_specs.xlsx")
flight_data = load_flight_data("data/flight_test.csv")

# Process single observation
row = flight_data[1, :]
runway_spec = get_runway_spec(runway_db, row.airport_runway)
corners = extract_corners(row)

# Estimate pose
pose = estimate_pose_6dof(runway_spec, corners)
println("Estimated position: $(pose.position)")
println("Estimated orientation: $(pose.orientation)")
```

### Batch Processing with Integrity Monitoring
```julia
# Process entire dataset
results = map(eachrow(flight_data)) do row
    runway_spec = get_runway_spec(runway_db, row.airport_runway)
    corners = extract_corners(row)
    uncertainties = extract_uncertainties(row)
    
    # Estimate with uncertainty
    pose_dist = estimate_pose_with_uncertainty(
        runway_spec, corners, uncertainties
    )
    
    # Check integrity
    raim_stat = compute_raim_statistic(pose_dist, runway_spec, corners)
    integrity_ok = check_integrity(raim_stat)
    
    (pose=pose_dist, integrity=integrity_ok, raim_statistic=raim_stat)
end

# Analyze results
integrity_performance = analyze_integrity_performance(
    results, flight_data.confusion_matrix_value
)
```

## File Structure
```
RunwayPoseEstimation.jl/
├── src/
│   ├── RunwayPoseEstimation.jl      # Main module file
│   ├── data_management/
│   │   ├── runway_database.jl       # Runway data loading
│   │   └── flight_data.jl          # Flight test data processing
│   ├── coordinate_systems/
│   │   ├── types.jl                # Coordinate system types
│   │   └── transformations.jl      # Coordinate transformations
│   ├── camera_model/
│   │   ├── projection.jl           # Camera projection functions
│   │   └── calibration.jl          # Camera parameter handling
│   ├── pose_estimation/
│   │   ├── optimization.jl         # Nonlinear least squares
│   │   ├── six_dof.jl             # 6-DOF pose estimation
│   │   └── three_dof.jl           # 3-DOF position estimation
│   ├── uncertainty_quantification/
│   │   ├── noise_models.jl         # Pixel uncertainty modeling
│   │   ├── propagation.jl          # Uncertainty propagation
│   │   └── calibration.jl          # Calibration assessment
│   ├── integrity_monitoring/
│   │   ├── raim.jl                 # RAIM implementation
│   │   ├── statistics.jl           # Test statistics
│   │   └── performance.jl          # Performance metrics
│   └── visualization/
│       ├── plotting.jl             # Basic plotting functions
│       ├── diagnostics.jl          # Diagnostic visualizations
│       └── analysis.jl             # Analysis and reporting
├── lib/
│   └── ProbabilisticParameterEstimators/  # Uncertainty quantification
├── test/
│   ├── runtests.jl
│   ├── unit/                       # Unit tests
│   ├── integration/                # Integration tests
│   └── data/                       # Test data
├── examples/
│   ├── basic_usage.jl
│   ├── batch_processing.jl
│   └── integrity_analysis.jl
├── docs/
│   ├── make.jl
│   └── src/
└── Project.toml
```

## Development Roadmap

### Phase 1: Core Infrastructure
- [x] Coordinate system types and transformations
- [x] Camera model and projection functions
- [x] Basic pose estimation (6-DOF and 3-DOF)
- [x] Data loading utilities

### Phase 2: Uncertainty Quantification
- [x] Integration with ProbabilisticParameterEstimators
- [x] Noise model construction
- [ ] Uncertainty propagation
- [ ] Calibration assessment tools

### Phase 3: Integrity Monitoring
- [ ] RAIM statistic computation
- [ ] Chi-squared test implementation
- [ ] Performance metrics and analysis
- [ ] Batch processing capabilities

### Phase 4: Visualization and Analysis
- [ ] Plotting and visualization tools
- [ ] Diagnostic capabilities
- [ ] Performance analysis utilities
- [ ] Documentation and examples

### Phase 5: Optimization and Testing
- [ ] Performance optimization
- [ ] Comprehensive test suite
- [ ] Documentation completion
- [ ] Package registration

## Python Usage

### Installation
For now, grab the latest package bundle from the [*Package Python* workflow](https://github.com/RomeoV/RunwayLib.jl/actions/workflows/package-python.yml) by clicking on the most recent run, downloading the artifact and unzipping it.

``` sh
unzip poseest-python-package.zip
```

Then you can install the library with

``` python
pip install -e poseest
```

e.g. using `uv`

``` sh
uv venv
uv pip install -e poseest
uv run python
```


### Usage

The compiled library can be used from Python:

```python
from poseest import (
    estimate_pose_3dof, estimate_pose_6dof, WorldPoint, ProjectionPoint, Rotation,
    CameraConfig, DiagonalCovariance
)

# Define runway corners in world coordinates (meters)
runway_corners = [
    WorldPoint(0.0, -50.0, 0.0),      # near left
    WorldPoint(0.0, 50.0, 0.0),       # near right
    WorldPoint(1000.0, -50.0, 0.0),   # far left
    WorldPoint(1000.0, 50.0, 0.0)     # far right
]

# Observed projections in image coordinates (pixels)
projections = [
    ProjectionPoint(320.0, 240.0),
    ProjectionPoint(380.0, 240.0),
    ProjectionPoint(380.0, 280.0),
    ProjectionPoint(320.0, 280.0)
]

# Diagonal covariance (pixel variances, not std deviations)
pixel_variances = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
covariance = DiagonalCovariance(variances=pixel_variances)

# 6-DOF estimation (position + orientation)
result_6dof = estimate_pose_6dof(
    runway_corners=runway_corners,
    projections=projections,
    camera_config=CameraConfig.OFFSET,
    covariance=covariance
)

# 3-DOF estimation (position only, known orientation)
known_rotation = Rotation(yaw=-0.01, pitch=0.1, roll=0.02)
result_3dof = estimate_pose_3dof(
    runway_corners=runway_corners,
    projections=projections,
    known_rotation=known_rotation,
    camera_config=CameraConfig.OFFSET,
    covariance=covariance
)

print(f"6-DOF: position={result_6dof.position}, rotation={result_6dof.rotation}")
print(f"3-DOF: position={result_3dof.position}")
```

For full covariance matrices, use `FullCovariance(matrix=cov_8x8)` instead of `DiagonalCovariance`.

## Contributing

This library is designed to be modular and extensible. Key extension points include:

- **New pose estimation algorithms**: Implement additional optimization methods
- **Alternative uncertainty methods**: Add new uncertainty quantification approaches  
- **Enhanced integrity monitoring**: Implement advanced RAIM variants
- **Additional visualizations**: Create new plotting and analysis tools

Each module should maintain clear interfaces and comprehensive documentation to facilitate contributions and extensions.
