# Integrity Check

In our paper [valentin2025predictive](@cite) we introduce an "integrity check" that lets us compare the reprojection error of the estimated pose to the magnitude of the predicted uncertainties by adopting an algorithm similar to RAIM [joerger2014solution](@cite).
This figure provides a brief illustration, with more details in [valentin2025predictive](@cite).

![An illustration of the integrity check.](figs/raim_explanation.svg)

```@docs; canonical = false
compute_integrity_statistic
```

### Usage Example

```@example
using RunwayLib, Unitful.DefaultSymbols, Rotations, LinearAlgebra, Distributions

# Define runway corners in world coordinates
runway_corners = [
    WorldPoint(0.0m, 50m, 0m),     # near left
    WorldPoint(3000.0m, 50m, 0m),  # far left
    WorldPoint(3000.0m, -50m, 0m), # far right
    WorldPoint(0.0m, -50m, 0m),    # near right
]

# True camera pose
cam_pos = WorldPoint(-2000.0m, 12m, 150m)
cam_rot = RotZYX(roll=1.5°, pitch=5°, yaw=0°)

# Project corners to image (with small noise for realism)
observed_corners = [project(cam_pos, cam_rot, pt) for pt in runway_corners]

# Create point features with noise model
noise_model = UncorrGaussianNoiseModel([Normal(0, 2.0) for _ in 1:8])
pf = PointFeatures(runway_corners, observed_corners, CAMERA_CONFIG_OFFSET, noise_model)

# Compute integrity statistic (points only)
result = compute_integrity_statistic(cam_pos, cam_rot, pf)
println("Integrity statistic: $(result.stat)")
println("P-value: $(result.p_value)")
println("Degrees of freedom: $(result.dofs)")
```

When observations match the model well (low noise, correct pose), we expect a high p-value.
A low p-value (e.g., < 0.05) suggests the measurements are inconsistent with the pose estimate.

#### With Line Features

The integrity check also supports line observations for increased redundancy:

```@example
using RunwayLib, Unitful.DefaultSymbols, Rotations, LinearAlgebra

runway_corners = [
    WorldPoint(0.0m, 50m, 0m),
    WorldPoint(3000.0m, 50m, 0m),
    WorldPoint(3000.0m, -50m, 0m),
    WorldPoint(0.0m, -50m, 0m),
]
cam_pos = WorldPoint(-2000.0m, 12m, 150m)
cam_rot = RotZYX(roll=1.5°, pitch=5°, yaw=0°)

# Point features
observed_corners = [project(cam_pos, cam_rot, pt) for pt in runway_corners]
pf = PointFeatures(runway_corners, observed_corners)

# Line features (runway edges)
line_endpoints = [
    (runway_corners[1], runway_corners[2]),  # left edge
    (runway_corners[3], runway_corners[4]),  # right edge
]
observed_lines = [
    let p1 = project(cam_pos, cam_rot, ep[1]), p2 = project(cam_pos, cam_rot, ep[2])
        getline(p1, p2)
    end
    for ep in line_endpoints
]
lf = LineFeatures(line_endpoints, observed_lines)

# Compute integrity with both points and lines
result = compute_integrity_statistic(cam_pos, cam_rot, pf, lf)
println("With lines - DOFs: $(result.dofs), p-value: $(result.p_value)")
```

## Worst-Case Fault Analysis
If a set of measurements and estimated pose passes the integrity check, we next want to determine the **protection level**, which is the maximum deviation in pose from our estimate that could go undetected. This is computed using worst-case fault direction and failure mode slope analysis, which identifies the measurement error patterns most likely to cause unacceptably large positioning errors without triggering the integrity check.

### Usage Example

```@example
using RunwayLib, Unitful.DefaultSymbols, Rotations, LinearAlgebra

runway_corners = [
    WorldPoint(0.0m, 50m, 0m),     # near left
    WorldPoint(3000.0m, 50m, 0m),  # far left
    WorldPoint(3000.0m, -50m, 0m),  # far right
    WorldPoint(0.0m, -50m, 0m),    # near right
]

cam_pos = WorldPoint(-2000.0m, 12m, 150m)
cam_rot = RotZYX(roll=1.5°, pitch=5°, yaw=0°)

noise_level = 2.0
sigmas = noise_level * ones(length(runway_corners))
noise_cov = Diagonal(repeat(sigmas .^ 2, inner=2))

H = RunwayLib.compute_H(cam_pos, cam_rot, runway_corners)

alpha_idx = 3           # monitor height error
fault_indices = [1, 2]  # assume faults in first two image measurements
H=H[:, 1:3]             # position-only Jacobian

# Compute worst-case undetected fault subset impact on monitored height
f_dir, g_slope = RunwayLib.compute_worst_case_fault_direction_and_slope(
    alpha_idx,
    fault_indices,
    H,
    noise_cov,
)
```

### API Reference

```@docs; canonical = false
compute_worst_case_fault_direction_and_slope
```

### Theory
This section formalizes the worst-case fault analysis used to compute protection levels following a successful integrity check.

#### Worst-case fault direction
The **worst-case fault direction** ``\mathbf{f}_i`` is the measurement error
pattern within a specified fault subset that maximizes error in the monitored
parameter while remaining undetectable by the integrity test. Following Joerger
et al. [joerger2014solution](@cite), we compute it as:

```math
\mathbf{f}_i = \mathbf{A}_i (\mathbf{A}_i^T \mathbf{P} \mathbf{P}^T \mathbf{A}_i)^{-1} (\mathbf{A}_i^T \mathbf{s}_0)
```

#### Failure mode slope
The **failure mode slope** ``g`` quantifies how rapidly the monitored parameter error grows relative to the detection statistic along the worst-case fault direction. Larger values of ``g`` indicate greater vulnerability: small undetected faults can induce large state estimation errors. It is defined as the ratio of the squared mean estimate error to the non-centrality parameter of the test statistic:

```math
g^2 = \mathbf{s}_0^T \mathbf{A}_i (\mathbf{A}_i^T \mathbf{P} \mathbf{P}^T \mathbf{A}_i)^{-1} \mathbf{A}_i^T \mathbf{s}_0
```

#### Definitions
- ``\mathbf{H}`` is the Jacobian matrix relating changes in measurements to changes in pose parameters
- ``\mathbf{P} = (\mathbf{I} - \mathbf{H}(\mathbf{H}^T \mathbf{H})^{-1}\mathbf{H}^T) * L^{-1}`` is the parity projection matrix
- ``\mathbf{\Sigma} = LL^T`` is the noise covariance
- ``\mathbf{A}_i`` is the fault selection matrix corresponding to the measurement indices specified by `fault_indices`
- ``\mathbf{s}_0`` is the extraction vector for the monitored parameter specified by `alpha_idx`

The subscript ``i`` identifies a specific **fault subset** (specified by `fault_indices`) and monitored **parameter** (specified by `alpha_idx`).

```@bibliography
Pages = ["integrity_check.md"]
Canonical = true
```
