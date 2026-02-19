# Solution Separation Protection Levels

After a set of measurements passes the [integrity check](@ref Integrity-Check), we want to bound the maximum position error that could go undetected. The **solution separation** method [joerger2014solution](@cite) computes this bound — the **protection level** — by comparing the all-in-view pose estimate to subset estimates, each obtained by excluding one keypoint from the observation set.

This page walks through a complete protection level computation for all three position axes (along-track, cross-track, altitude). For background on the integrity check and worst-case fault analysis, see [Integrity Check](@ref).

## Definitions

The following quantities are used throughout this computation. Equation references are to Joerger et al. [joerger2014solution](@cite).

**Geometry and estimation:**
- ``\mathbf{H}`` — Jacobian matrix (``n_\text{obs} \times 6``), relating measurement perturbations to pose parameter changes. Computed via `RunwayLib.compute_H`.
- ``\mathbf{W} = \boldsymbol{\Sigma}^{-1}`` — Weight matrix (inverse of the measurement noise covariance).
- ``\mathbf{P}_0 = (\mathbf{H}^\top \mathbf{W} \mathbf{H})^{-1}`` — All-in-view state covariance (``6 \times 6``).
- ``\boldsymbol{\alpha}`` — Extraction vector selecting the monitored position component, e.g. ``\boldsymbol{\alpha} = [1,0,0,0,0,0]^\top`` for along-track.
- ``\sigma_0^2 = \boldsymbol{\alpha}^\top \mathbf{P}_0 \, \boldsymbol{\alpha}`` — All-in-view variance of the monitored parameter.

**Fault hypotheses:**
- ``n_\text{kp}`` — Number of keypoints (each is a potential single-fault source).
- ``h = n_\text{kp}`` — Number of fault hypotheses (one per keypoint for single-keypoint faults).
- ``p_\text{fault}`` — Prior probability that any given keypoint is faulty.
- ``P_{H_0} = (1 - p_\text{fault})^{n_\text{kp}}`` — Prior probability that no keypoint is faulty.

**Risk budgets (see RTCA DO-253D, ICAO Annex 10):**
- ``I_\text{REQ}`` — Total integrity risk requirement.
- ``C_\text{REQ}`` — Total continuity risk requirement.
- ``I_{\text{alloc},k} = I_\text{REQ} / (h+1)`` — Per-hypothesis integrity allocation (equal allocation across ``h`` fault hypotheses + 1 fault-free hypothesis).
- ``C_{\text{alloc},k} = C_\text{REQ} / h`` — Per-hypothesis continuity allocation.

**Per-hypothesis quantities (hypothesis ``k`` = remove keypoint ``k``):**
- ``\mathbf{H}_k``, ``\mathbf{W}_k`` — Jacobian and weight matrix after removing the two observation rows corresponding to keypoint ``k``.
- ``\mathbf{P}_k = (\mathbf{H}_k^\top \mathbf{W}_k \mathbf{H}_k)^{-1}`` — Subset covariance.
- ``\sigma_k^2 = \boldsymbol{\alpha}^\top \mathbf{P}_k \, \boldsymbol{\alpha}`` — Subset variance.
- ``\sigma_{\Delta k}^2 = \sigma_k^2 - \sigma_0^2`` — Solution separation variance.

**Thresholds and protection levels (Joerger Eq. 58):**
- ``K_\text{FA} = \Phi^{-1}\!\bigl(1 - C_{\text{alloc},k} / (2 P_{H_0})\bigr)`` — False alarm quantile (from continuity budget).
- ``K_\text{md} = \Phi^{-1}\!\bigl(1 - I_{\text{alloc},k} / (2 p_\text{fault})\bigr)`` — Missed detection quantile (from integrity budget).
- ``\text{PL}_k = K_\text{FA} \cdot \sigma_{\Delta k} + K_\text{md} \cdot \sigma_k`` — Per-hypothesis protection level.
- ``\text{PL}_0 = K_{\text{md},0} \cdot \sigma_0`` — Fault-free protection level.
- ``\text{PL} = \max\!\bigl(\text{PL}_0, \max_k \text{PL}_k\bigr)`` — Overall protection level.

where ``\Phi^{-1}`` is the inverse of the standard normal CDF.

## Walkthrough

### Scenario setup

We define a standard runway with 4 corner keypoints, a camera pose, and project the corners with Gaussian pixel noise (``\sigma = 2\,\text{px}``).

```@example solsep
using RunwayLib, Unitful.DefaultSymbols, Rotations
import RunwayLib: compute_H, covmatrix
using Distributions, LinearAlgebra

runway_corners = [
    WorldPoint(0.0m, 50m, 0m),
    WorldPoint(3000.0m, 50m, 0m),
    WorldPoint(3000.0m, -50m, 0m),
    WorldPoint(0.0m, -50m, 0m),
]

cam_pos = WorldPoint(-2000.0m, 12m, 150m)
cam_rot = RotZYX(roll=1.5°, pitch=5°, yaw=0°)

true_observations = [project(cam_pos, cam_rot, p) for p in runway_corners]
noisy_observations = [p + ProjectionPoint(2.0*randn(2)px) for p in true_observations]
nothing # hide
```

### All-in-view pose estimate

We estimate the 6-DOF pose from all 4 keypoints, yielding 8 observations (2 pixel coordinates per keypoint) and 6 unknowns (3 position + 3 orientation).

```@example solsep
noise_model = UncorrGaussianNoiseModel([Normal(0, 2.0) for _ in 1:8])
pf = PointFeatures(runway_corners, noisy_observations, CAMERA_CONFIG_OFFSET, noise_model)
(cam_pos_est, cam_rot_est) = estimatepose6dof(pf)[(:pos, :rot)]
nothing # hide
```

### All-in-view covariance

The all-in-view covariance ``\mathbf{P}_0 = (\mathbf{H}^\top \mathbf{W} \mathbf{H})^{-1}`` captures the estimation uncertainty when all observations are used.

```@example solsep
H = compute_H(cam_pos_est, cam_rot_est, pf)   # 8×6 Jacobian
W = inv(covmatrix(noise_model))                 # 8×8 weight matrix
P₀ = inv(H' * W * H)                           # 6×6 state covariance
nothing # hide
```

### Risk budgets and fault priors

We define the integrity and continuity risk requirements and the prior fault probability. With ``h+1`` total hypotheses (``h`` faults + 1 fault-free), the risk budget is allocated equally.

```@example solsep
n_kp = 4                  # number of keypoints
h = n_kp                  # single-keypoint fault hypotheses
p_fault = 1e-3            # prior: P(keypoint k is faulty)
P_H0 = (1-p_fault)^n_kp  # prior: P(fault-free)
I_REQ = 1e-5              # total integrity risk budget
C_REQ = 1e-3              # total continuity risk budget

# Equal allocation across hypotheses
I_alloc_0 = I_REQ / (h+1)          # fault-free slice
I_alloc_k = I_REQ / (h+1)          # per-fault slice
C_alloc_k = C_REQ / h              # per-fault continuity slice
nothing # hide
```

### Per-hypothesis protection levels

For each fault hypothesis ``k``, we remove keypoint ``k`` (rows ``2k{-}1`` and ``2k`` from ``\mathbf{H}`` and ``\mathbf{W}``), recompute the subset covariance, and derive the protection level per axis. We compute this for all three position components: along-track (index 1), cross-track (index 2), and altitude (index 3).

```@example solsep
axes = [(1, "Along-track"), (2, "Cross-track"), (3, "Altitude")]

PLs_per_axis = map(axes) do (axis_idx, _)
    α = zeros(6); α[axis_idx] = 1.0
    σ²₀ = α' * P₀ * α

    PLs = map(1:n_kp) do k
        # Remove keypoint k → remove rows (2k-1, 2k) from H and W
        keep = vcat([2i-1:2i for i in 1:n_kp if i != k]...)

        Hₖ = H[keep, :]
        Wₖ = W[keep, keep]

        # Subset covariance Pₖ = (Hₖᵀ Wₖ Hₖ)⁻¹
        Pₖ = inv(Hₖ' * Wₖ * Hₖ)

        σ²ₖ  = α' * Pₖ * α           # subset solution variance
        σ²_Δk = σ²ₖ - σ²₀             # solution separation variance

        σ_Δk = sqrt(max(σ²_Δk, 0.0))  # guard against numerical noise
        σₖ   = sqrt(σ²ₖ)

        # Joerger Eq (58): threshold from continuity budget
        K_FA = quantile(Normal(), 1 - C_alloc_k / (2P_H0))

        # Integrity quantile
        K_md = quantile(Normal(), 1 - I_alloc_k / (2p_fault))

        PL_k = K_FA * σ_Δk + K_md * σₖ
    end

    # Fault-free term
    σ₀ = sqrt(α' * P₀ * α)
    K_md_0 = quantile(Normal(), 1 - I_alloc_0 / (2P_H0))
    PL_0 = K_md_0 * σ₀

    # Overall protection level for this axis
    max(PL_0, maximum(PLs))
end
nothing # hide
```

### Results

The overall protection level for each position axis:

```@example solsep
println("| Axis | Protection Level (m) |")
println("|------|---------------------|")
for ((_, name), pl) in zip(axes, PLs_per_axis)
    println("| $name | $(round(pl; digits=2)) |")
end
```

```@bibliography
Pages = ["solution_separation.md"]
Canonical = false
```
