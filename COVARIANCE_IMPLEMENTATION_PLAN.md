# Covariance Matrix Implementation Plan

## Overview
This document outlines the implementation plan for supporting flexible covariance matrix specifications in RunwayLib. The goal is to support multiple ways to specify noise/covariance for keypoints while maintaining backward compatibility.

## Current State Analysis

### Existing Noise Model System
- Uses `ProbabilisticParameterEstimators` package with `UncorrGaussianNoiseModel` and `CorrGaussianNoiseModel`
- `covmatrix(noisemodel)` returns covariance matrix (currently can be `Diagonal` type)
- In `optimization.jl`, `PoseOptimizationParams6DOF/3DOF` stores `Linv = inv(cholesky(cov).U')` 
- Optimization uses `Linv * errors` to weight the residuals

### Current C API Structure
- Functions accept pointer arrays for runway corners and projections
- No current noise/covariance specification in C API
- Uses hard-coded default noise models

### Current Python Interface
- Wraps C API functions
- No current covariance specification options

## Implementation Plan

### Phase 1: Julia Core Infrastructure

#### 1.1 Update Optimization Infrastructure (`src/pose_estimation/optimization.jl`)
**Objective:** Ensure preallocated caches use dense matrices for Linv

**Changes:**
- In `PoseOptimizationParams6DOF` and `PoseOptimizationParams3DOF` constructors:
  - Wrap `Linv = inv(cholesky(cov).U')` with `Matrix()` to ensure dense storage
  - Update type annotation from `M <: AbstractMatrix{T′′}` to `M <: Matrix{T′′}` if needed
- Update cache constants `CACHE_6DOF` and `CACHE_3DOF` to use dense matrices
- Verify that `_defaultnoisemodel` produces appropriate dense matrices

#### 1.2 Define Covariance Type Enum (`src/c_api.jl`)
**Objective:** Create enum for different covariance specification methods

**New Types:**
```julia
@cenum COVARIANCE_TYPE_C::Cint begin
    COV_SCALAR = 0          # Single noise value for all keypoints/directions
    COV_DIAGONAL_FULL = 1   # Diagonal matrix (length = 2*n_keypoints)  
    COV_BLOCK_DIAGONAL = 2  # 2x2 matrix per keypoint (length = 4*n_keypoints)
    COV_FULL_MATRIX = 3     # Full covariance matrix (length = 4*n_keypoints^2)
end
```

### Phase 2: C API Enhancement

#### 2.1 Update C API Functions (`src/c_api.jl`)
**Objective:** Add covariance specification to pose estimation functions

**New Function Signatures:**
```julia
Base.@ccallable function estimate_pose_6dof_with_covariance(
    runway_corners_::Ptr{WorldPointF64},
    projections_::Ptr{ProjectionPointF64}, 
    num_points::Cint,
    covariance_data::Ptr{Cdouble},
    covariance_type::COVARIANCE_TYPE_C,
    camera_config::CAMERA_CONFIG_C,
    result::Ptr{PoseEstimate_C}
)::Cint
```

**Implementation Details:**
- Parse `covariance_data` based on `covariance_type` enum
- Construct appropriate `NoiseModel` from the parsed data
- Use existing optimization infrastructure with the custom noise model
- Maintain backward compatibility by keeping original functions that use default noise

#### 2.2 Covariance Data Interpretation Logic
**Objective:** Convert C pointer data to Julia NoiseModel based on type

**Covariance Types:**
1. **COV_SCALAR**: `covariance_data[0]` = noise standard deviation for all measurements
2. **COV_DIAGONAL_FULL**: `covariance_data[0..2*n-1]` = diagonal variances 
3. **COV_BLOCK_DIAGONAL**: `covariance_data[0..4*n-1]` = 2x2 covariance matrices (stored row-major)
4. **COV_FULL_MATRIX**: `covariance_data[0..4*n^2-1]` = full covariance matrix (stored row-major)

### Phase 3: Python Interface Enhancement

#### 3.1 Add Covariance Classes (`poseest_py/poseest/core.py`)
**Objective:** Provide Python classes for covariance specification

**New Classes:**
```python
@dataclass
class CovarianceSpec:
    """Base class for covariance specifications."""
    pass

@dataclass  
class ScalarCovariance(CovarianceSpec):
    noise_std: float

@dataclass
class DiagonalCovariance(CovarianceSpec):
    variances: List[float]  # Length = 2*n_keypoints

@dataclass
class BlockDiagonalCovariance(CovarianceSpec):
    block_matrices: List[List[List[float]]]  # List of 2x2 matrices

@dataclass
class FullCovariance(CovarianceSpec):
    matrix: List[List[float]]  # 2n x 2n matrix
```

#### 3.2 Update Python API Functions
**Objective:** Add covariance parameters to pose estimation functions

**Updated Signatures:**
```python
def estimate_pose_6dof(
    runway_corners: List[WorldPoint],
    projections: List[ProjectionPoint],
    camera_config: CameraConfig = CameraConfig.OFFSET,
    covariance: Optional[CovarianceSpec] = None
) -> PoseEstimate
```

#### 3.3 C Interface Integration
**Objective:** Convert Python covariance specs to C API calls

**Implementation:**
- Add covariance conversion functions to transform Python classes to C arrays
- Update ctypes function signatures to include covariance parameters  
- Implement fallback to default noise model when covariance is None

### Phase 4: Testing Infrastructure

#### 4.1 Julia Tests (`test/unit/test_covariance_specification.jl`)
**Objectives:**
- Test all covariance type parsing from C pointers
- Verify optimization works with different covariance specifications
- Test edge cases (singular matrices, negative values, etc.)
- Performance tests comparing dense vs sparse matrices

#### 4.2 Python Tests (`poseest_py/tests/test_covariance.py`) 
**Objectives:**
- Test covariance class creation and validation
- Test integration with C API
- Test all covariance types produce reasonable results
- Test backward compatibility

#### 4.3 Integration Tests
**Objectives:**
- End-to-end tests with realistic covariance specifications
- Compare results between different covariance representations of same uncertainty
- Test compilation and library rebuilding workflow

### Phase 5: Documentation and Examples

#### 5.1 Usage Examples
- Document covariance specification options in docstrings
- Create example scripts showing different covariance use cases
- Update existing examples to optionally use custom covariance

#### 5.2 API Documentation
- Update function documentation for new parameters
- Document covariance type enum values and expected data formats
- Provide guidance on choosing appropriate covariance specifications

## Implementation Order

1. **Julia optimization.jl updates** - Foundation for dense matrix support
2. **C API enum and covariance parsing** - Core functionality 
3. **C API function updates** - New function signatures
4. **Python covariance classes** - User-facing API
5. **Python C integration** - Connect Python to C API
6. **Julia testing** - Verify core functionality
7. **Python testing** - Verify end-to-end workflow
8. **Integration testing** - Full system validation

## Backward Compatibility

- Original C API functions remain unchanged and use default noise models
- Python functions accept optional covariance parameter (None = use defaults)
- Existing Julia code continues to work with current NoiseModel system

## Success Criteria

- [ ] All covariance types can be specified from Python
- [ ] C API correctly interprets covariance data based on enum
- [ ] Optimization produces consistent results across covariance representations
- [ ] Python tests pass after library recompilation
- [ ] Julia tests pass with new covariance specifications
- [ ] Performance is acceptable with dense matrices
- [ ] Backward compatibility maintained