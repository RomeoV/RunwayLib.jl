# Integrity Check

In our paper [valentin2025predictive](@cite) we introduce an "integrity check" that let's us compare the reprojection error of the estimated pose to the magnitude of the predicted uncertainties by adoping an algorithm similar to RAIM [joerger2014solution](@cite).
This figure provides a brief illustration, with more details in [valentin2025predictive](@cite).

![An illustration of the integrity check.](figs/raim_explanation.svg)

```@docs; canonical = false
compute_integrity_statistic
```

## Worst-Case Fault Analysis

We support the computation of worst-case fault direction and failure mode slope for analyzing integrity risk. This analysis helps quantify the impact of measurement faults on pose estimation accuracy.

### Theory

The **worst-case fault direction** ``\mathbf{f}_i`` is the measurement error pattern that maximizes integrity risk, which is the probability of undetected faults causing unacceptably large positioning errors. Following [joerger2014solution](@cite), we compute it as:

```math
\mathbf{f}_i = \mathbf{A}_i (\mathbf{A}_i^T \mathbf{P} \mathbf{A}_i)^{-1} (\mathbf{A}_i^T \mathbf{s}_0)
```

The **failure mode slope** ``g`` quantifies the sensitivity of a monitored parameter to faults in the worst-case direction. It is defined as the ratio of the squared mean estimate error to the non-centrality parameter of the test statistic:

```math
g^2 = \mathbf{s}_0^T \mathbf{A}_i (\mathbf{A}_i^T \mathbf{P} \mathbf{A}_i)^{-1} \mathbf{A}_i^T \mathbf{s}_0
```

where the subscript ``i`` identifies a specific **fault subset** (specified by `fault_indices`) and monitored **parameter** (specified by `alpha_idx`):
- ``\mathbf{P} = \mathbf{I} - \mathbf{H}(\mathbf{H}^T \mathbf{H})^{-1}\mathbf{H}^T`` is the parity projection matrix
- ``\mathbf{A}_i`` is the fault selection matrix that selects the measurements in `fault_indices` (e.g., if `fault_indices = [1,3]`, then ``\mathbf{A}_i`` picks out measurements 1 and 3)
- ``\mathbf{s}_0`` is the extraction vector for the parameter `alpha_idx` (e.g., if `alpha_idx = 3` for z-position, then ``\mathbf{s}_0`` extracts the z-component from the state estimate)
- ``\mathbf{H}`` is the Jacobian matrix relating measurement errors to pose parameter errors

Thus ``\mathbf{f}_i`` represents the worst-case fault direction **for the specific combination of fault subset and monitored parameter**.

### API Reference

```@docs; canonical = false
compute_worst_case_fault_direction_and_slope
```

```@bibliography
Pages = ["integrity_check.md"]
Canonical = true
```
