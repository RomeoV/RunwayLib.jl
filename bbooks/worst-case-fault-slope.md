# New Book

```julia (editor=true, logging=false, output=true)
using RunwayLib
using WGLMakie
```
```julia (editor=true, logging=false, output=true)
using RunwayLib, Unitful.DefaultSymbols, Rotations
import RunwayLib: px

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

cam_pos_est
```
```julia (editor=true, logging=false, output=true)
H = RunwayLib.compute_H(cam_pos, cam_rot, runway_corners)
```
```julia (editor=true, logging=false, output=true)
using LinearAlgebra
Qt = nullspace(H')
Q = Qt'
```
```julia (editor=true, logging=false, output=true)
p = 100*rand(2)
q = Qt * p
noisy_observations_with_error = noisy_observations + [
    ProjectionPoint(q[i], q[i+1])px for i in 1:2:length(q)
]
(cam_pos_est2, cam_rot_est2) = estimatepose6dof(
    PointFeatures(runway_corners, noisy_observations_with_error)
)[(:pos, :rot)]
cam_pos_est - cam_pos_est2
```
```julia (editor=true, logging=false, output=true)
# find fi = (In - HH^+)^{inv}* alpha H^+
function computefi(alphaidx, H; normalize=true)
    α = let alpha = zeros(6);  alpha[alphaidx] = 1; alpha end
    pinvH = pinv(H)
    s_0 = pinvH' * α
    f_i = (I - H * pinvH) \ s_0# |> normalize
    normalize && normalize!(f_i)
    f_i
end
f_i = computefi(1, H)
g_fi = sqrt(s_0' * computefi(1, H; normalize=false))
```
```julia (editor=true, logging=false, output=true)
# find fi = (In - HH^+)^{inv}* alpha H^+
alpha_ = 2
α = let alpha = zeros(6);  alpha[alpha_] = 1; alpha end
pinvH = pinv(H)
s_0 = pinvH' * α
f_i = (I - H * pinvH) \ s_0 |> normalize
```
```julia (editor=true, logging=false, output=true)
using BracketingNonlinearSolve
```
```julia (editor=true, logging=false, output=true)
compute_integrity_statistic(
    cam_pos, cam_rot,
    runway_corners,
    noisy_observations, 
    2.0*I(length(runway_corners)*2)
)
```
```julia (editor=true, logging=false, output=true)
function get_pvalue(ø, (; threshold, std, alphaidx, H))
    f = computefi(alphaidx, H)
    noisy_observations_with_error = noisy_observations .+ [
        ø*ProjectionPoint(f[i], f[i+1])px for i in 1:2:length(f)
    ]

    (cam_pos, cam_rot) = estimatepose6dof(
        PointFeatures(runway_corners, noisy_observations_with_error)
    )[(:pos, :rot)]

    p = compute_integrity_statistic(
        cam_pos, cam_rot,
        runway_corners,
        noisy_observations_with_error,
        std*I(length(runway_corners)*2)
    ).p_value
    p - threshold
end
```
```julia (editor=true, logging=false, output=true)
ps = (; threshold=0.05, std=2, alphaidx=1, H)
prob = IntervalNonlinearProblem(get_pvalue, [0.0, 10_000.0], ps)
ø = solve(prob).u
```
```julia (editor=true, logging=false, output=true)
get_pvalue(0, ps), get_pvalue(10_000, ps)
```
```julia (editor=true, logging=false, output=true)
let    
    (; alphaidx, H) = ps
    f = computefi(alphaidx, H)
    noisy_observations_with_error = noisy_observations .+ [
        ø*ProjectionPoint(f[i], f[i+1])px for i in 1:2:length(f)
    ]

    (cam_pos, cam_rot) = estimatepose6dof(
        PointFeatures(runway_corners, noisy_observations_with_error)
    )[(:pos, :rot)]
    cam_pos[alphaidx] - cam_pos_est[alphaidx]
end
```
```julia (editor=true, logging=false, output=true)
ps = (; threshold=0.05, std=2, alphaidx=1, H)
prob = IntervalNonlinearProblem(get_pvalue, [-10_000.0, 0.0], ps)
ø = solve(prob).u
```
```julia (editor=true, logging=false, output=true)
let    
    (; alphaidx, H) = ps
    f = computefi(alphaidx, H)
    noisy_observations_with_error = noisy_observations .+ [
        ø*ProjectionPoint(f[i], f[i+1])px for i in 1:2:length(f)
    ]

    (cam_pos, cam_rot) = estimatepose6dof(
        PointFeatures(runway_corners, noisy_observations_with_error)
    )[(:pos, :rot)]
    cam_pos[alphaidx] - cam_pos_est[alphaidx]
end
```
```julia (editor=true, logging=false, output=true)
ps = (; threshold=0.05, std=2, alphaidx=1, H)
cam_pos_guesses = map(([-10_000.0, 0.0], [0.0, 10_000.0])) do interval
    (; H, alphaidx) = ps
    prob = IntervalNonlinearProblem(get_pvalue, interval, ps)
    ø = solve(prob).u
    
    f = computefi(alphaidx, H)
    noisy_observations_with_error = noisy_observations .+ [
        ø*ProjectionPoint(f[i], f[i+1])px for i in 1:2:length(f)
    ]

    (cam_pos, cam_rot) = estimatepose6dof(
        PointFeatures(runway_corners, noisy_observations_with_error)
    )[(:pos, :rot)]
    cam_pos
end
```
```julia (editor=true, logging=false, output=true)
function getworstpoints(alphaidx)
    ps = (; threshold=0.05, std=2, alphaidx, H)
cam_pos_guesses = map(([-10_000.0, 0.0], [0.0, 10_000.0])) do interval
    (; H, alphaidx) = ps
    prob = IntervalNonlinearProblem(get_pvalue, interval, ps)
    ø = solve(prob).u
    
    f = computefi(alphaidx, H)
    noisy_observations_with_error = noisy_observations .+ [
        ø*ProjectionPoint(f[i], f[i+1])px for i in 1:2:length(f)
    ]

    (cam_pos, cam_rot) = estimatepose6dof(
        PointFeatures(runway_corners, noisy_observations_with_error)
    )[(:pos, :rot)]
    cam_pos
end
end
```
```julia (editor=true, logging=false, output=true)
pose_guesses = map(1:1_000) do _
    noisy_observations3 = [p + ProjectionPoint(2.0*randn(2)px) for p in true_observations]

    (cam_pos_est3, cam_rot_est3) = estimatepose6dof(
        PointFeatures(runway_corners, noisy_observations3)
    )[(:pos, :rot)]

    passed = compute_integrity_statistic(
        cam_pos_est3, cam_rot_est3,
        runway_corners,
        noisy_observations3, 
        2.0*I(length(runway_corners)*2)
    ).p_value > 0.05
    (; cam_pos=cam_pos_est3, cam_rot=cam_rot_est3, passed)
end;
```
```julia (editor=true, logging=false, output=true)
import RunwayLib: _ustrip
blue, orange, _ = Makie.wong_colors()
fig = Figure(; size=(800, 800));
ax = Axis3(fig[1,1])
slider = Makie.Slider(fig[2,1], range=1:3, startvalue=1)
pts = @lift getworstpoints($(slider.value))
scatter!(ax, (@lift [
    (cam_pos .|> _ustrip(m)),
    ($(pts)[1] .|> _ustrip(m)),
    ($(pts)[2] .|> _ustrip(m))
]); color=[orange, blue, blue], markersize=20)
scatter!(ax, [p.cam_pos .|> _ustrip(m) for p in pose_guesses]; color=[p.passed ? :green : :purple for p in pose_guesses], alpha=0.3)
fig
```
