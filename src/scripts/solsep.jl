using RunwayLib, Unitful.DefaultSymbols, Rotations
import RunwayLib: compute_H, covmatrix
using Distributions

runway_corners = [
    WorldPoint(0.0m, 50m, 0m),     # near left
    WorldPoint(3000.0m, 50m, 0m),  # far left
    WorldPoint(3000.0m, -50m, 0m),  # far right
    WorldPoint(0.0m, -50m, 0m),    # near right
]

cam_pos = WorldPoint(-2000.0m, 12m, 150m)
cam_rot = RotZYX(roll=1.5°, pitch=5°, yaw=0°)

true_observations = [project(cam_pos, cam_rot, p) for p in runway_corners]
noisy_observations = [p + ProjectionPoint(2.0*randn(2)px) for p in true_observations]

(cam_pos_est, cam_rot_est) = estimatepose6dof(
    PointFeatures(runway_corners, noisy_observations)
)[(:pos, :rot)]

x_ks = map(1:4) do neg_idx
    idx = [i!= neg_idx for i in 1:4]
    # idx = ntuple(i->i !=neg_idx, Val(4))
    wpts = runway_corners[idx]
    obs = noisy_observations[idx]
    noise_model = UncorrGaussianNoiseModel(
        [Normal(0, 2.0) for _ in 1:6]
    )
    pf = PointFeatures(wpts, obs, CAMERA_CONFIG_OFFSET, noise_model)

    (cam_pos_est, cam_rot_est) = estimatepose6dof(pf)

    # H = compute_H(cam_pos_est, cam_rot_est, pf)
    # noise_model = UncorrGaussianNoiseModel(
    #     [Normal(0, 2.0) for _ in 1:9]
    # )
    # Σ_eps = covmatrix(noise_model)
    # Σ_beta = cov(inv(H) * MvNormal(Σ_eps))
    cam_pos_est.x
end

x_0 = cam_pos_est.x
σ_ss = std(x_ks, mean=x_0)

noise_model = UncorrGaussianNoiseModel(
    [Normal(0, 2.0) for _ in 1:8]
)
pf = PointFeatures(runway_corners, noisy_observations, CAMERA_CONFIG_OFFSET, noise_model)
H = compute_H(cam_pos_est, cam_rot_est, pf)
Σ_eps = covmatrix(noise_model)
Σ_beta = inv(cov(H' * MvNormal(inv(Σ_eps))))  # propagate uncertainty
σ_0 = sqrt(Σ_beta[1, 1])*m
fa_rate = 0.02  # false alarm rate
fm_rate = 1e-5  # false miss rate
K_FA = quantile(Normal(), 1.0-0.02)
K_I = quantile(Normal(), 1.0-1e-5)
K_FA * σ_ss + K_I * sqrt(σ_ss^2 + σ_0^2)
