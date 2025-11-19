### A Pluto.jl notebook ###
# v0.20.20

using Markdown
using InteractiveUtils

# ╔═╡ b5b8f3c8-c4dc-11f0-82e6-e3e1218a8fd8
import Pkg; Pkg.activate(".")

# ╔═╡ 47423636-18d6-42cb-85e6-4a0909dc168d
begin
	using RunwayLib
	using WGLMakie
	using BracketingNonlinearSolve
	using Unitful.DefaultSymbols, Rotations
	using Unitful
    import RunwayLib: px, _ustrip
	using LinearAlgebra
end

# ╔═╡ 46af6473-88bf-49b9-8dc9-0a72e995f784
html"""<style>
main {
    max-width: 1000px;
}
"""

# ╔═╡ ddee502b-6245-45eb-b4cc-ce4a4f749fcf
runway_corners = [
    WorldPoint(0.0m, 50m, 0m),     # near left
    WorldPoint(3000.0m, 50m, 0m),  # far left
    WorldPoint(3000.0m, -50m, 0m),  # far right
    WorldPoint(0.0m, -50m, 0m),    # near right
]

# ╔═╡ 2fe79916-81bf-4487-bd21-4656783cc4c6
cam_pos = WorldPoint(-2000.0m, 12m, 150m)

# ╔═╡ 020be658-c6dc-48a3-abb8-34cf8b1fd449
cam_rot = RotZYX(roll=1.5°, pitch=5°, yaw=0°)

# ╔═╡ 951488bb-bf5a-434e-8251-8664ca58ee7d
true_observations = [project(cam_pos, cam_rot, p) for p in runway_corners]

# ╔═╡ b377ee57-1d61-4787-bb9d-ed2b760ef23d
noisy_observations = [p + ProjectionPoint(2.0*randn(2)px) for p in true_observations]

# ╔═╡ c6a57e0f-50c7-461a-a6e8-9281991b9e44
(cam_pos_est, cam_rot_est) = estimatepose6dof(
    PointFeatures(runway_corners, noisy_observations)
)[(:pos, :rot)]

# ╔═╡ b027b7ad-098e-4048-97d1-f4ce311c5ac4
H = RunwayLib.compute_H(cam_pos, cam_rot, runway_corners)

# ╔═╡ 128eddd5-1607-4188-b824-747c34ad5572
Q = let
	Qt = nullspace(H')
	Qt'
end

# ╔═╡ 36e1df5f-9cf1-43d1-a2a4-9e63c56ae7c8
cycle(xs::AbstractVector) = xs[[eachindex(xs); first(eachindex(xs))]]

# ╔═╡ d5a57f78-dae7-4ceb-aab5-a83a6dddc204
(x, y) = true_observations[1]

# ╔═╡ 49755fa4-babc-43a7-86db-2afcc96b18f7
Makie.wong_colors();

# ╔═╡ 56e87bb5-b7e6-45c1-a2cb-95e0aaf2a000
drand = normalize(randn(8))

# ╔═╡ a530d3e4-22db-4744-8b39-5947be16772c
with_theme(theme_black()) do
	c1, c2, c3, c4, c5, c6, c7 = Makie.wong_colors()

	fig = Figure(; size=(900, 600))
	slg = Makie.SliderGrid(
		fig[2, 1:2],
		(label="perturbation dir 1: ", range=-100:0.5:100, startvalue=0.0),
		(label="perturbation dir 2:", range=-100:0.5:100, startvalue=0.0),
		(label="random pertubation: ", range=-100:0.5:100, startvalue=0.0),
	)
	sl1, sl2, sl3 = slg.sliders;
	yperturb = @lift [Q' drand] * [$(sl1.value); $(sl2.value); $(sl3.value)]

	perturbed_observations = @lift noisy_observations .+ ProjectionPoint.(eachcol(reshape($yperturb, 2, :)))*px
	cam_pos_pert = @lift estimatepose6dof(
	    PointFeatures(runway_corners, $perturbed_observations)
	)[:pos]

	ax3 = Axis3(fig[1,1]; title="Pose Estimate")
	scatter!(ax3, @lift [
		              cam_pos_est .|> _ustrip(m),
		              $(cam_pos_pert) .|> _ustrip(m)
				  ]; color=[c1, c6])

	
	ax = Axis(fig[1,2]; yreversed=true, title="Perturbed Observations")
	
	scatterlines!(ax, [obs .|> _ustrip(px) for obs in cycle(true_observations)])

	arrows2d!(ax, Point2.([obs .|> _ustrip(px) for obs in true_observations]),
			    @lift(Point2.(eachcol(reshape($yperturb, 2, :))));
			  color=:red
		   )
			
	fig
end

# ╔═╡ 853b6a37-c9aa-4ef3-9a46-bcd4e06807f7
# find fi = (In - HH^+)^{inv}* alpha H^+
function computefi(alphaidx, H; normalize=true)
    α = let alpha = zeros(6);  alpha[alphaidx] = 1; alpha end
    pinvH = pinv(H)
    s_0 = pinvH' * α
    f_i = (I - H * pinvH) \ s_0# |> normalize
    normalize && normalize!(f_i)
    f_i
end
#f_i = computefi(1, H)
#g_fi = sqrt(s_0' * computefi(1, H; normalize=false))

# ╔═╡ 6e3438b3-81b1-4055-87f9-28731bd674c2
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

# ╔═╡ 17bffe68-fcf8-406d-b2e1-31ebdece6af3
function getworstpoints(alphaidx)
    ps = (; threshold=0.05, std=2, alphaidx, H)
    cam_pos_guesses = map(
        ([-10_000.0, 0.0], 
         [0.0, 10_000.0] )
	) do interval
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

# ╔═╡ 3beb4274-3955-4ac4-af4c-2770e58a769b
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

# ╔═╡ 470b7e5f-73b5-4df3-9d89-67b3a67a8eb1
let
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
end

# ╔═╡ Cell order:
# ╟─46af6473-88bf-49b9-8dc9-0a72e995f784
# ╠═b5b8f3c8-c4dc-11f0-82e6-e3e1218a8fd8
# ╠═47423636-18d6-42cb-85e6-4a0909dc168d
# ╠═ddee502b-6245-45eb-b4cc-ce4a4f749fcf
# ╠═2fe79916-81bf-4487-bd21-4656783cc4c6
# ╠═020be658-c6dc-48a3-abb8-34cf8b1fd449
# ╠═951488bb-bf5a-434e-8251-8664ca58ee7d
# ╠═b377ee57-1d61-4787-bb9d-ed2b760ef23d
# ╠═c6a57e0f-50c7-461a-a6e8-9281991b9e44
# ╠═b027b7ad-098e-4048-97d1-f4ce311c5ac4
# ╠═128eddd5-1607-4188-b824-747c34ad5572
# ╠═36e1df5f-9cf1-43d1-a2a4-9e63c56ae7c8
# ╠═d5a57f78-dae7-4ceb-aab5-a83a6dddc204
# ╠═49755fa4-babc-43a7-86db-2afcc96b18f7
# ╠═56e87bb5-b7e6-45c1-a2cb-95e0aaf2a000
# ╠═a530d3e4-22db-4744-8b39-5947be16772c
# ╠═853b6a37-c9aa-4ef3-9a46-bcd4e06807f7
# ╠═6e3438b3-81b1-4055-87f9-28731bd674c2
# ╠═17bffe68-fcf8-406d-b2e1-31ebdece6af3
# ╠═3beb4274-3955-4ac4-af4c-2770e58a769b
# ╠═470b7e5f-73b5-4df3-9d89-67b3a67a8eb1
