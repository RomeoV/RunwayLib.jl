# Static Array Removal Implementation Plan

## Goal
Remove static arrays from pose estimation to allow flexibility in number of keypoints, while keeping static arrays where they provide performance benefits for small, fixed-size data structures.

## What Should REMAIN as Static Arrays
- `WorldPoint`, `CameraPoint`, `ProjectionPoint` - These are FieldVectors with 2-3 elements, perfect for static arrays
- Small rotation vectors (3 elements)
- Small error vectors within individual computations
- Any fixed-size mathematical operations

## What Should BECOME Regular Arrays
- Collections of points (runway_corners, observed_corners, projected_corners)
- Cache initialization vectors (u0 for optimization)
- Any data structure that needs to vary in size based on number of keypoints

## Current Issues Found

### 1. C API Issues
- `RotYPRF64` changed from `SVector{3,Float64}` to `Vector{Float64}` - this breaks C interop
- The C API expects fixed-size data structures for interop
- **Fix**: Keep `RotYPRF64` as `SVector{3,Float64}` but handle conversions properly

### 2. Type Mismatches
- Some functions expect static arrays but now receive regular arrays
- Need to be careful about where conversions happen

### 3. Test Failures
- C API tests failing due to type mismatches with rotation vectors
- Need to fix the pointer conversions

## Implementation Strategy

### Phase 1: Fix Core Pose Estimation (DONE)
- ✅ Remove MVector from cache initialization  
- ✅ Change collections of points to regular arrays
- ✅ Update error vector concatenation

### Phase 2: Fix C API (IN PROGRESS)
- 🔄 Keep rotation types as SVector for C interop
- 🔄 Fix unsafe_convert issues
- 🔄 Ensure point collections work with variable sizes

### Phase 3: Update Tests
- 🔄 Fix C API test failures
- ⏸️ Add tests for varying keypoint numbers (8, 16)

### Phase 4: Cleanup
- ⏸️ Remove forked LinearSolve dependency
- ⏸️ Move const caches to function scope if needed

## Specific Files to Modify

### src/pose_estimation/optimization.jl
- ✅ Remove `using StaticArrays: MVector`
- ✅ Change cache initialization from MVector to regular arrays
- ✅ Update error vector creation
- ✅ Change default parameters from SA[...] to [...]

### src/c_api.jl  
- 🔄 **REVERT** RotYPRF64 back to SVector{3,Float64}
- 🔄 Fix array conversions for point collections
- 🔄 Handle variable-size point arrays properly

### test files
- ✅ Change test data from SA[...] to [...]
- 🔄 Fix C API test pointer issues

### src/precompile_workloads.jl
- ✅ Change runway_corners from SA[...] to [...]

## Next Steps
1. Fix RotYPRF64 type back to SVector for C compatibility
2. Fix unsafe_convert issues in C API tests
3. Test with varying numbers of keypoints
4. Clean up dependencies