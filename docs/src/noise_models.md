# Noise Models

Noise models describe the measurement uncertainty in observed image features (point coordinates and line parameters).
They are used throughout RunwayLib for:
- **Pose estimation**: weighting the residuals so that noisier measurements contribute less to the cost function.
- **Integrity monitoring**: normalizing residuals for the chi-squared consistency test (see [Integrity Check](@ref)).
- **Worst-case fault analysis**: computing protection levels that account for the measurement noise covariance.

RunwayLib re-exports the noise model types from [`ProbabilisticParameterEstimators.jl`](https://github.com/RomeoV/ProbabilisticParameterEstimators.jl):

- [`NoiseModel`](@extref ProbabilisticParameterEstimators.NoiseModel) — abstract supertype
- [`UncorrGaussianNoiseModel`](@extref ProbabilisticParameterEstimators.UncorrGaussianNoiseModel) — independent Gaussian noise per measurement (or per 2D/3D group)
- [`CorrGaussianNoiseModel`](@extref ProbabilisticParameterEstimators.CorrGaussianNoiseModel) — a single multivariate normal capturing cross-measurement correlations

Every noise model supports `covmatrix` which returns the full covariance matrix.

## Choosing a Noise Model

| Situation | Recommended type | Example |
|---|---|---|
| All measurements have the same, independent uncertainty | `UncorrGaussianNoiseModel` with scalar `Normal` distributions | Default (σ = 2 px) |
| Each measurement has its own independent uncertainty | `UncorrGaussianNoiseModel` with per-measurement `Normal` distributions | Heteroscedastic detector |
| Each keypoint has correlated x/y uncertainty | `UncorrGaussianNoiseModel` with per-keypoint `MvNormal` distributions | Detector with covariance output |
| Measurements are correlated across keypoints | `CorrGaussianNoiseModel` with a full `MvNormal` | Correlated image processing pipeline |

## Default Noise Model

When no noise model is provided, RunwayLib uses an `UncorrGaussianNoiseModel` with `Normal(0, 2.0)` per coordinate.
For point features this means 2 distributions per keypoint (x and y), and for line features 3 distributions per observation (r, cos θ, sin θ).

## Usage Examples

### Uncorrelated noise — same σ for all measurements

```@example noise
using RunwayLib, Unitful.DefaultSymbols, Rotations, Distributions

runway_corners = [
    WorldPoint(0.0m, 50m, 0m),
    WorldPoint(3000.0m, 50m, 0m),
    WorldPoint(3000.0m, -50m, 0m),
    WorldPoint(0.0m, -50m, 0m),
]

cam_pos = WorldPoint(-2000.0m, 12m, 150m)
cam_rot = RotZYX(roll=1.5°, pitch=5°, yaw=0°)
observations = [project(cam_pos, cam_rot, pt) for pt in runway_corners]

# 4 keypoints × 2 coordinates = 8 scalar Normal distributions
σ = 3.0  # pixels
noise_model = UncorrGaussianNoiseModel([Normal(0.0, σ) for _ in 1:8])

pf = PointFeatures(runway_corners, observations, CAMERA_CONFIG_OFFSET, noise_model)
(; pos, rot) = estimatepose6dof(pf)
pos
```

### Uncorrelated noise — per-keypoint 2×2 covariance

When the detector reports a 2×2 covariance for each keypoint, use `MvNormal` distributions:

```@example noise
using LinearAlgebra

# Each keypoint gets its own 2×2 covariance matrix
keypoint_covs = [
    [4.0 0.1; 0.1 4.0],
    [9.0 0.2; 0.2 9.0],
    [6.25 0.0; 0.0 6.25],
    [7.84 -0.1; -0.1 7.84],
]
noise_model_2x2 = UncorrGaussianNoiseModel(
    [MvNormal(zeros(2), Σ) for Σ in keypoint_covs]
)

pf2 = PointFeatures(runway_corners, observations, CAMERA_CONFIG_OFFSET, noise_model_2x2)
(; pos, rot) = estimatepose6dof(pf2)
pos
```

### Correlated noise — full covariance matrix

If measurements are correlated across keypoints (e.g., due to shared image processing), use `CorrGaussianNoiseModel`:

```@example noise
n = 8  # 4 keypoints × 2 coordinates
Σ_full = Matrix{Float64}(4.0I, n, n)
# Add correlation between x-coords of adjacent keypoints
Σ_full[1, 3] = Σ_full[3, 1] = 0.5
Σ_full[3, 5] = Σ_full[5, 3] = 0.5
Σ_full[5, 7] = Σ_full[7, 5] = 0.5

noise_model_corr = CorrGaussianNoiseModel(MvNormal(zeros(n), Σ_full))

pf3 = PointFeatures(runway_corners, observations, CAMERA_CONFIG_OFFSET, noise_model_corr)
(; pos, rot) = estimatepose6dof(pf3)
pos
```

### Noise models with line features

Line features use 3 residual components per observation (r, cos θ, sin θ), so the noise model has 3 distributions per line:

```@example noise
line_pts = [
    (runway_corners[1], runway_corners[2]),  # left edge
    (runway_corners[3], runway_corners[4]),  # right edge
]
observed_lines = [
    let p1 = project(cam_pos, cam_rot, ep[1]),
        p2 = project(cam_pos, cam_rot, ep[2])
        getline(p1, p2)
    end
    for ep in line_pts
]

# 2 lines × 3 components = 6 distributions
line_noise = UncorrGaussianNoiseModel([Normal(0.0, 2.0) for _ in 1:6])
lf = LineFeatures(line_pts, observed_lines, CAMERA_CONFIG_OFFSET, line_noise)

(; pos, rot) = estimatepose6dof(
    PointFeatures(runway_corners, observations),
    lf
)
pos
```

## How Noise Models Are Used Internally

When a [`PointFeatures`](@ref) or [`LineFeatures`](@ref) struct is constructed with a noise model,
the covariance matrix ``\Sigma`` is extracted via `covmatrix` and its Cholesky factor ``L`` is computed (``\Sigma = L L^\top``).
The pose optimization objective then whitens the residual vector ``r`` as

```math
r_{\text{weighted}} = L^{-1} r
```

so that the nonlinear least-squares solver minimizes ``\| L^{-1} r \|^2``, which is the Mahalanobis distance.
This ensures that measurements with larger uncertainty contribute proportionally less to the fit.

## C API Covariance Specification

When calling RunwayLib from C (or other languages via the C API), covariance data is passed as a raw pointer together with a `COVARIANCE_TYPE` enum:

| Enum value | Meaning | Data layout |
|---|---|---|
| `COV_DEFAULT` | Use the default noise model (σ = 2 px) | Pointer may be null |
| `COV_SCALAR` | Single σ for all measurements | 1 `Float64` |
| `COV_DIAGONAL_FULL` | Per-coordinate variances | `2n` `Float64` values |
| `COV_BLOCK_DIAGONAL` | Per-keypoint 2×2 covariance | `4n` `Float64` values (column-major 2×2 blocks) |
| `COV_FULL_MATRIX` | Full covariance matrix | `(2n)²` `Float64` values (column-major) |

See `parse_covariance_data` in `src/c_api.jl` for the implementation.

## API Reference

```@docs; canonical = false
PointFeatures
LineFeatures
```
