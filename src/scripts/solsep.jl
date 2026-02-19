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

# --- All-in-view estimate ---
noise_model = UncorrGaussianNoiseModel([Normal(0, 2.0) for _ in 1:8])
pf = PointFeatures(runway_corners, noisy_observations, CAMERA_CONFIG_OFFSET, noise_model)
(cam_pos_est, cam_rot_est) = estimatepose6dof(pf)[(:pos, :rot)]

# --- All-in-view covariance P₀ = (Hᵀ W H)⁻¹ ---
H = compute_H(cam_pos_est, cam_rot_est, pf)   # 8×6 Jacobian
W = inv(covmatrix(noise_model))                 # 8×8 weight matrix
P₀ = inv(H' * W * H)                           # 6×6 state covariance

# State of interest: x-position (adjust index to your state ordering)
α = zeros(6); α[1] = 1.0
σ²₀ = α' * P₀ * α

# --- Requirements ---
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

# --- Per-hypothesis protection levels ---
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

    # Integrity quantile: P(|εₖ| > PL-T) ≤ I_alloc_k / p_fault
    K_md = quantile(Normal(), 1 - I_alloc_k / (2p_fault))

    PL_k = K_FA * σ_Δk + K_md * σₖ

    (; k, PL_k, σ_Δk, σₖ, K_FA, K_md)
end

# Fault-free term: just bounding nominal error
K_md_0 = quantile(Normal(), 1 - I_alloc_0 / (2P_H0))
PL_0 = K_md_0 * sqrt(σ²₀)

# Overall protection level
PL = max(PL_0, maximum(pl.PL_k for pl in PLs))
