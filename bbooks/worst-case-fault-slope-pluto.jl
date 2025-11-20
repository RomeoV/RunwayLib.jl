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
	using BracketingNonlinearSolve  # line search
	using Unitful.DefaultSymbols, Rotations
	using Unitful
    import RunwayLib: px, _ustrip
	using LinearAlgebra
	import RunwayLib.StaticArrays: Size
	using SparseArrays
	using FileIO, MeshIO  # load airplane mesh
	using Tau  # τ=2π
end

# ╔═╡ 46af6473-88bf-49b9-8dc9-0a72e995f784
html"""<style>
main {
    max-width: 1000px;
}
"""

# ╔═╡ 64d2c0fd-2542-4b2c-80f6-134ed8434c3b
HTML("""
<!-- the wrapper span -->
<div>
	<button id="myrestart" href="#">Restart Notebook</button>
	
	<script>
		const div = currentScript.parentElement
		const button = div.querySelector("button#myrestart")
		const cell= div.closest('pluto-cell')
		console.log(button);
		button.onclick = function() { restart_nb() };
		function restart_nb() {
			console.log("Restarting Notebook");
		        cell._internal_pluto_actions.send(                    
		            "restart_process",
                            {},
                            {
                                notebook_id: editor_state.notebook.notebook_id,
                            }
                        )
		};
	</script>
</div>
""")

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

# ╔═╡ 59a0ab1e-0360-4d24-9320-fb3966062b9d
aircraft_model = load(joinpath("assets", "A320NeoV2_lowpoly.stl"));

# ╔═╡ 120a3051-4909-4e65-a35d-82e76b706567
function setup_corner_selections(fig)
	gl = GridLayout(fig[3, 1:2], tellwidth = false)		
	axismenu = Menu(gl[1, 0], 
					options = [
						"alongtrack", "crosstrack", "altitude",
						"yaw", "pitch", "roll",
					],
					default = "alongtrack", tellwidth=true, width=100)
	
	subgl = GridLayout(gl[1, 1])
	
	cb1 = Checkbox(subgl[1, 1], checked = true)
	cb2 = Checkbox(subgl[2, 1], checked = false)
	cb3 = Checkbox(subgl[3, 1], checked = false)
	cb4 = Checkbox(subgl[4, 1], checked = false)
	cbs = [cb1, cb2, cb3, cb4]

	Label(subgl[1, 2], "Near Left", halign = :left)
	Label(subgl[2, 2], "Far Left", halign = :left)
	Label(subgl[3, 2], "Far Right", halign = :left)
	Label(subgl[4, 2], "Near Right", halign = :left)
	rowgap!(subgl, 8)
	colgap!(subgl, 8)
	(; gl, cbs, axismenu)
end

# ╔═╡ e2850d53-5bd7-4515-8ccf-1f8b3cb4f02a
Rotations.QuatRotation(RotZ(τ*rad/4)) |> Rotations.params

# ╔═╡ 684bd932-6cd1-4fe8-bd1f-76998de1b2e7
function quat_from_rotmatrix(dcm::AbstractMatrix{T}) where {T<:Real}
    a2 = 1 + dcm[1,1] + dcm[2,2] + dcm[3,3]
    a = sqrt(a2)/2
    b,c,d = (dcm[3,2]-dcm[2,3])/4a, (dcm[1,3]-dcm[3,1])/4a, (dcm[2,1]-dcm[1,2])/4a
    return Quaternion(a,b,c,d)
end

# ╔═╡ b495605a-ffe7-4783-a490-1d635731da0a
# corrects the mesh rotation, adds our rotation, and turns it to quaternion for Makie
function to_corrected_quat(R::Rotation{3})
	quat = QuatRotation(R * RotZ(1/4 * τ))
	Quaternion(quat.x, quat.y, quat.z, quat.w)
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

# ╔═╡ 683af497-4a84-40dd-87e8-0e06ba04d6a3
function computefi(alphaidx, fault_indices, H; normalize=true)
    α = let alpha = zeros(6); alpha[alphaidx] = 1; alpha end
    S_0 = pinv(H)
    s_0 = S_0' * α
    
    # Create selection matrix for dual fault
    n = size(H, 1)
	nf = length(fault_indices)
	A_i = sparse(collect(fault_indices), 1:nf, ones(nf), n, nf)
    # Compute worst-case fault direction
    proj_parity = I - H * S_0
    f_i = A_i * ((A_i' * proj_parity * A_i) \ (A_i' * s_0))
    
    normalize && normalize!(f_i)
    f_i
end


# ╔═╡ a530d3e4-22db-4744-8b39-5947be16772c
with_theme(theme_black()) do
	c1, c2, c3, c4, c5, c6, c7 = Makie.wong_colors()

	fig = Figure(; size=(900, 600))
	slg = Makie.SliderGrid(
		fig[2, 1:2],
		(label="perturbation dir 1: ", range=-100:0.5:100, startvalue=0.0),
		(label="perturbation dir 2:", range=-100:0.5:100, startvalue=0.0),
		(label="random pertubation: ", range=-100:0.5:100, startvalue=0.0),
		(label="worst case pertubation: ", range=-100:0.5:100, startvalue=0.0),
	)
	sl1, sl2, sl3, sl4 = slg.sliders;

	(; gl, cbs, axismenu) = setup_corner_selections(fig)
	cb1, cb2, cb3, cb4 = cbs
	fi_indices = @lift let
		idx = findall([$(cb1.checked), $(cb2.checked), $(cb3.checked), $(cb4.checked)])
		[(1:2:8)[idx];
		 (2:2:8)[idx]]
	end
	alphaidx = @lift findfirst(==($(axismenu.selection)), $(axismenu.options))
	yobs_pts = Point2.(true_observations)
	yperturb = @lift Q' * [$(sl1.value); $(sl2.value)]
	yperturb_pts = @lift ProjectionPoint.(eachcol(reshape($yperturb, 2, :)))*px
	yrand_pts = @lift ProjectionPoint.(eachcol(reshape($(sl3.value) * drand, 2, :)))*px
	f_i = @lift computefi($alphaidx, $fi_indices, H)
	yfi_pts = @lift ProjectionPoint.(eachcol(reshape($(sl4.value) * $f_i, 2, :)))*px

	perturbed_observations = @lift noisy_observations .+ $yperturb_pts .+ $yrand_pts .+ $yfi_pts
	cam_pose_est_pert = @lift estimatepose6dof(
	    PointFeatures(runway_corners, $perturbed_observations)
	)
	cam_pos_pert = @lift $(cam_pose_est_pert).pos
	cam_rot_pert = @lift $(cam_pose_est_pert).rot

	passed = @lift compute_integrity_statistic(
        $(cam_pose_est_pert)[(:pos, :rot)]...,
        runway_corners,
        $perturbed_observations, 
        2.0*I(length(runway_corners)*2)
    ).p_value > 0.05

	ax3 = Axis3(fig[1,1]; title="Pose Estimate")#, limits=(-2500, 1500, -100, 100, 0, 200))
	meshscatter!(ax3, @lift [
		              cam_pos_est .|> _ustrip(m),
		              $(cam_pos_pert) .|> _ustrip(m)
				  ]; 
				 color=@lift([(c1, 1.0), (($(passed) ? :green : :red), 1.0)]),
				 marker=aircraft_model, 
				 markersize=1/3,
				 rotation=@lift to_corrected_quat.([
					 $cam_rot_est, $cam_rot_pert
				 ])			 
				)

	
	ax = Axis(fig[1,2]; yreversed=true, title="Perturbed Observations")
	
	scatterlines!(ax, [obs .|> _ustrip(px) for obs in cycle(true_observations)])
    scatterlines!(ax, @lift([obs .|> _ustrip(px) for obs in cycle(yobs_pts .+ $yperturb_pts .+ $yrand_pts .+ $yfi_pts)]); color=(:yellow, 0.5), linestyle=:dash)
	
	arrows2d!(ax, [obs .|> _ustrip(px) for obs in yobs_pts],
			    yperturb_pts;
			  color=:red
		   )
	arrows2d!(ax, @lift([obs .|> _ustrip(px) for obs in (yobs_pts .+ $yperturb_pts)]),
			    yrand_pts;
			  color=c7
		   )
	arrows2d!(ax, @lift([obs .|> _ustrip(px) for obs in (yobs_pts .+ $yperturb_pts .+ $yrand_pts)]),
			yfi_pts;
		  color=c4
	   )


	#arrows2d!(ax, @lift(Point2.([obs .|> _ustrip(px) for obs in true_observations]) .+ $yperturb_pts),
	#		yf_ipts;
	#	  color=c4
	#   )

			
	fig
end

# ╔═╡ 656fdbdc-2614-4aa0-a4dc-4021c5b2ea8c
f_i = computefi(1, H)

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
# ╟─64d2c0fd-2542-4b2c-80f6-134ed8434c3b
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
# ╠═59a0ab1e-0360-4d24-9320-fb3966062b9d
# ╠═120a3051-4909-4e65-a35d-82e76b706567
# ╠═e2850d53-5bd7-4515-8ccf-1f8b3cb4f02a
# ╠═684bd932-6cd1-4fe8-bd1f-76998de1b2e7
# ╠═b495605a-ffe7-4783-a490-1d635731da0a
# ╠═a530d3e4-22db-4744-8b39-5947be16772c
# ╠═656fdbdc-2614-4aa0-a4dc-4021c5b2ea8c
# ╠═853b6a37-c9aa-4ef3-9a46-bcd4e06807f7
# ╠═683af497-4a84-40dd-87e8-0e06ba04d6a3
# ╠═6e3438b3-81b1-4055-87f9-28731bd674c2
# ╠═17bffe68-fcf8-406d-b2e1-31ebdece6af3
# ╠═3beb4274-3955-4ac4-af4c-2770e58a769b
# ╠═470b7e5f-73b5-4df3-9d89-67b3a67a8eb1
