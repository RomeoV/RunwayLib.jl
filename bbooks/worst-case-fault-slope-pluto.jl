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
	import Rotations: params
	using Unitful
    import RunwayLib: px, _ustrip
	using LinearAlgebra
	import RunwayLib.StaticArrays: Size
	using SparseArrays
	using FileIO, MeshIO  # load airplane mesh
	using Tau  # τ=2π
	import Random
	using Distributions
	using UnitfulLinearAlgebra
end

# ╔═╡ 46af6473-88bf-49b9-8dc9-0a72e995f784
html"""<style>
main {
    max-width: 75%;
}
pluto-editor main {
    align-self: center;
    margin-right: 0;
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

# ╔═╡ 3fae75fa-7760-4878-b97f-888ed1dbbf0f
px_std = sqrt(2)  # this corresponds to the default noise model which has px_var = 2

# ╔═╡ b377ee57-1d61-4787-bb9d-ed2b760ef23d
noisy_observations = let; Random.seed!(1)
	[p + ProjectionPoint(px_std*randn(2)px) for p in true_observations]
end

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

# ╔═╡ 49755fa4-babc-43a7-86db-2afcc96b18f7
Makie.wong_colors();

# ╔═╡ 56e87bb5-b7e6-45c1-a2cb-95e0aaf2a000
drand = normalize(randn(8))

# ╔═╡ 59a0ab1e-0360-4d24-9320-fb3966062b9d
aircraft_model = load(joinpath("assets", "A320NeoV2_lowpoly.stl"));

# ╔═╡ 120a3051-4909-4e65-a35d-82e76b706567
function setup_corner_selections(figpos)
	gl = GridLayout(figpos, tellwidth = false)
	Label(gl[1, -1], text="Worst Case Axis:")
	axismenu = Menu(gl[1, 0], 
					options = [
						"alongtrack", "crosstrack", "altitude",
						"roll", "pitch", "yaw",
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

# ╔═╡ b495605a-ffe7-4783-a490-1d635731da0a
# corrects the mesh rotation, adds our rotation, and turns it to quaternion for Makie
function to_corrected_quat(R::Rotation{3})
	quat = QuatRotation(R * RotZ(1/4 * τ))
	Quaternion(quat.x, quat.y, quat.z, quat.w)
end

# ╔═╡ 683af497-4a84-40dd-87e8-0e06ba04d6a3
function computefi(alphaidx, fault_indices, H; normalize=true, sixdof=true)
    α = let alpha = zeros(sixdof ? 6 : 3); alpha[alphaidx] = 1; alpha end
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

	fig = Figure(; size=(1200, 600))
	slg = Makie.SliderGrid(
		fig[3, 1:2],
		(label="perturbation dir 1: ", range=-100:0.5:100, startvalue=0.0),
		(label="perturbation dir 2:", range=-100:0.5:100, startvalue=0.0),
		(label="random pertubation: ", range=-100:0.5:100, startvalue=0.0),
		(label="worst case pertubation: ", range=-10:0.05:10, startvalue=0.0),
	)
	sl1, sl2, sl3, sl4 = slg.sliders;

	control_menu = GridLayout(fig[2, 1:2])

	(; gl, cbs, axismenu) = setup_corner_selections(control_menu[1, 1])
	cb1, cb2, cb3, cb4 = cbs
	fi_indices = @lift let
		idx = findall([$(cb1.checked), $(cb2.checked), $(cb3.checked), $(cb4.checked)])
		[(1:2:8)[idx];
		 (2:2:8)[idx]]
	end

	visual_menu = GridLayout(control_menu[1,2]; tell_width=false)
	Label(visual_menu[1, 1], text="Show Runway")
	show_runway = Checkbox(visual_menu[1, 2], checked=false)
	
	#alphaidx = @lift findfirst(==($(axismenu.selection)), $(axismenu.options))
	alphaidx = @lift Dict(1=>1, 2=>2, 3=>3, 4=>6, 5=>5, 6=>4)[$(axismenu.i_selected)]  # roll pitch yaw representation is reversed
	yobs_pts = Point2.(true_observations)
	yperturb = @lift Q' * [$(sl1.value); $(sl2.value)]
	yperturb_pts = @lift ProjectionPoint.(eachcol(reshape($yperturb, 2, :)))*px
	yrand_pts = @lift ProjectionPoint.(eachcol(reshape($(sl3.value) * drand, 2, :)))*px
	f_i = @lift computefi($alphaidx, $fi_indices, H)
	yfi_pts = @lift let sl4val = $(sl4.value)
		ProjectionPoint.(eachcol(reshape(sign(sl4val)*exp(abs(sl4val)) * $f_i, 2, :)))*px
	end

	perturbed_observations = @lift noisy_observations .+ $yperturb_pts .+ $yrand_pts .+ $yfi_pts
	cam_pose_est_pert = @lift estimatepose6dof(
	    PointFeatures(runway_corners, $perturbed_observations)
	)
	cam_pos_pert = @lift $(cam_pose_est_pert).pos
	cam_rot_pert = @lift $(cam_pose_est_pert).rot

	pose_delta_layout = GridLayout(control_menu[1,0])
	Label(pose_delta_layout[0, 0:1], text="Pose Delta", font=:bold)
	Label(pose_delta_layout[1, 0], text="x = \ny = \nz = ")
	Label(pose_delta_layout[1, 1], text=@lift(let 
		s = repr("text/plain", round.([typeof(1.0m)], (cam_pos_est - $cam_pos_pert); digits=2))
		split(s, '\n')[2:end] |> x->join(x, '\n')
	end), halign=:right)
	rowgap!(pose_delta_layout, 0)
	colgap!(pose_delta_layout, 0)
	Label(pose_delta_layout[0, 2:4], text="Attitude Delta", font=:bold)
	Label(pose_delta_layout[1, 2], text="roll\npitch\nyaw", justification=:left)
	Label(pose_delta_layout[1, 3], text="=\n=\n=", justification=:left)
	Label(pose_delta_layout[1, 4], text=@lift(let 
		s = repr("text/plain", round.([typeof(1.0°)], reverse(rad2deg.((params($cam_rot_pert) - params(cam_rot_est)).*rad)); digits=1))
		split(s, '\n')[2:end] |> x->join(x, '\n')
	end), halign=:right)


	passed = @lift compute_integrity_statistic(
        $(cam_pose_est_pert)[(:pos, :rot)]...,
        runway_corners,
        $perturbed_observations, 
        2.0*I(length(runway_corners)*2)
    ).p_value > 0.05

	# POSE VIEW
	ax3 = Axis3(fig[1,1]; title="Pose Estimate", 
				aspect=:data,
			    xlabel="x = alongtrack [m]",
			    ylabel="y = crosstrack [m]",
			    zlabel="z = altitude [m]"
			   )
	## Plot Aircraft
	meshargs = (;
		marker=aircraft_model, 
		markersize=@lift($(show_runway.checked) ? 3 : 1/3),
	)
	#=
	meshscatter!(ax3, @lift [
		              cam_pos_est .|> _ustrip(m),
		              $(cam_pos_pert) .|> _ustrip(m)
				  ]; 
				 color=@lift([(c1, 1.0), (($(passed) ? :green : :red), 1.0)]),
				 marker=aircraft_model, 
				 #markersize=1/3,
				 markersize=@lift($(show_runway.checked) ? 3 : 1/3),
				 rotation=@lift(to_corrected_quat.([
					 $cam_rot_est, $cam_rot_pert
				 ])),
				 label=["a", "b"],
				)
	=#
	meshscatter!(ax3, [cam_pos_est .|> _ustrip(m)];
			     color=c1, label="Reference", rotation=to_corrected_quat(cam_rot_est),
				 meshargs...)
	meshscatter!(ax3, @lift([$cam_pos_pert .|> _ustrip(m)]);
			     color=@lift((($(passed) ? :green : :red), 1.0)),
		         label="Perturbed", rotation=@lift(to_corrected_quat($cam_rot_pert)),
				 meshargs...)
	axislegend(ax3)
	on(cam_pos_pert) do _
		reset_limits!(ax3)
	end
	
	## Maybe show Runway
	poly!(ax3, [pt .|> _ustrip(m) for pt in cycle(runway_corners)], visible=show_runway.checked)
	on(show_runway.checked) do _; reset_limits!(ax3); end

	# PROJECTION VIEW
	ax = Axis(fig[1,2]; yreversed=true,
			  title="Perturbed Observations", aspect=DataAspect())
	## Plot original and perturbed runway projection
	scatterlines!(ax, [obs .|> _ustrip(px) for obs in cycle(true_observations)])
    scatterlines!(ax, @lift([obs .|> _ustrip(px) 
							 for obs in cycle(yobs_pts .+ $yperturb_pts .+ $yrand_pts .+ $yfi_pts)]);
				  color=(:yellow, 0.5), linestyle=:dash)

	## Plot various arrows, incrementally summing up
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
	on(yperturb_pts) do _; reset_limits!(ax); end
	on(yrand_pts) do _; reset_limits!(ax); end
	on(yfi_pts) do _; reset_limits!(ax); end
			
	fig
end

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

# ╔═╡ 532e4b44-f7f6-47b6-bb49-58257d2b1157
"""
Computes the Worst-Case Failure Mode Slope (g) for a given state and fault subset.
Ref: Equation (32) in Joerger et al. 2014.

g² = (s₀ᵀ A) (Aᵀ P A)⁻¹ (Aᵀ s₀)
"""
function compute_slope_nounits(alphaidx, fault_indices, H)
	# 1. Define extraction vector s₀ for the state of interest (alpha)
	α = let alpha = zeros(6); alpha[alphaidx] = 1; alpha end
	S_0 = pinv(H)
	s_0 = S_0' * α
	
	# 2. Define Parity Projection Matrix P
	# P = I - H(HᵀH)⁻¹Hᵀ = I - H S₀
	proj_parity = I - H * S_0

	# 3. Define Fault Selection Matrix A
	n = size(H, 1)
	nf = length(fault_indices)
	A_i = sparse(collect(fault_indices), 1:nf, ones(nf), n, nf)

	# 4. Compute Slope Squared (Eq 32)
	# The term m_Xi in the paper is Aᵀ s₀
	m_Xi = A_i' * s_0
	
	# The central term (Aᵀ (I - H S₀) A)⁻¹
	# This measures how "visible" faults in this subspace are to the parity check
	visibility_matrix = A_i' * proj_parity * A_i
	
	g_squared = m_Xi' * (visibility_matrix \ m_Xi)
	return sqrt(g_squared)
end


# ╔═╡ fe706472-f0f1-49cd-94db-ca4537012491
#LinearAlgebra.pinv(M::AbstractMatrix{<:Quantity}) = pinv(ustrip(M)) * inv(oneunit(first(M)))

# ╔═╡ a76ba211-2106-40b2-a326-cac15b4545c4
#LinearAlgebra.pinv(M::UnitfulMatrix) = pinv(M.data) * inv(oneunit(first(M)))

# ╔═╡ e00d78a4-0d5c-4e63-b0de-e2a5cac79abd
UnitfulMatrix(H * px/m) |> pinv

# ╔═╡ 05fd4f82-3e37-4f50-83ed-66474a8f3003
let H_=copy(H), alphaidx=1, fault_indices=[1, 2]
H = H_*px/m |> UnitfulMatrix
	# 1. Define extraction vector s₀ for the state of interest (alpha)
	α = let alpha = zeros(6); alpha[alphaidx] = 1; alpha end
	S_0 = pinv(H)
	s_0 = S_0' * α
	
	# 2. Define Parity Projection Matrix P
	# P = I - H(HᵀH)⁻¹Hᵀ = I - H S₀
	proj_parity = I - H * S_0

	# 3. Define Fault Selection Matrix A
	n = size(H, 1)
	nf = length(fault_indices)
	A_i = sparse(collect(fault_indices), 1:nf, ones(nf), n, nf) |> Matrix

	# 4. Compute Slope Squared (Eq 32)
	# The term m_Xi in the paper is Aᵀ s₀
	m_Xi = A_i' * s_0
	
	# The central term (Aᵀ (I - H S₀) A)⁻¹
	# This measures how "visible" faults in this subspace are to the parity check
	 	visibility_matrix = A_i' * proj_parity * A_i


end

# ╔═╡ 41d7ef29-0a65-488e-9a28-1d625c61703c
function compute_slope(alphaidx, fault_indices, H_)
	H = H_*px/m |> UnitfulMatrix
	# 1. Define extraction vector s₀ for the state of interest (alpha)
	α = let alpha = zeros(6); alpha[alphaidx] = 1; alpha end
	S_0 = pinv(H)
	s_0 = S_0' * α
	
	# 2. Define Parity Projection Matrix P
	# P = I - H(HᵀH)⁻¹Hᵀ = I - H S₀
	proj_parity = I - H * S_0

	# 3. Define Fault Selection Matrix A
	n = size(H, 1)
	nf = length(fault_indices)
	A_i = sparse(collect(fault_indices), 1:nf, ones(nf), n, nf) |> Matrix

	# 4. Compute Slope Squared (Eq 32)
	# The term m_Xi in the paper is Aᵀ s₀
	m_Xi = A_i' * s_0
	
	# The central term (Aᵀ (I - H S₀) A)⁻¹
	# This measures how "visible" faults in this subspace are to the parity check
	visibility_matrix = A_i' * proj_parity * A_i
	
	g_squared = m_Xi' * (visibility_matrix \ m_Xi)
	return sqrt(g_squared)
end


# ╔═╡ c208ce81-b980-405e-8698-f632e1898630
pinv(rand(4, 3))*1/m

# ╔═╡ f80aee26-b1d2-4183-aec8-2187f9c2cdfe
compute_slope(1, [1, 2], H)

# ╔═╡ ef5eb470-af13-4491-a5a4-d634e41bf6f6
begin
	alphaidx = 2
	fi_indices=[1, 2]
	# 1. Compute the Slope g [meters / pixel]
	# This tells us: for every 1 unit of detectable parity noise, how much position error occurs?
	slope_g = compute_slope(alphaidx, fi_indices, H)
	
	# 2. Determine the Detection Threshold (T)
	# The monitor checks if SSE < T². 
	# We need the T corresponding to our p-value requirement (alpha=0.05).
	dof = size(H, 1) - size(H, 2) # Degrees of Freedom = Measurements - States
	T_chisq = quantile(Chisq(dof), 0.95)
	
	# 3. Calculate Analytic Max Error
	# Max Error = Slope * Sigma * sqrt(Threshold)
	# Sigma = sqrt(2.0) because we passed 2.0*I as covariance
	sigma_val = sqrt(2.0)*px
	
	analytic_max_error = slope_g * sigma_val * sqrt(T_chisq)
	
	md"""
	### Analytic Results
	Based on Eq (32) and the Chi-Squared threshold:
	
	*   **Failure Mode Slope ($g_{Fi}$):** $(round(typeof(1.0*m/px), slope_g, digits=4))
	*   **$\chi^2$ Threshold ($T^2$):** $(round(T_chisq, digits=2))
	*   **Worst Case Error (Analytic):** $(round(typeof(1.0m), analytic_max_error, digits=4))
	"""
end


# ╔═╡ 58a21d5e-8a66-45b2-8202-aee966463df3
function estimate_pose_with_faultmagnitude(ø, (; alphaidx, fi_indices, H, noisy_observations))
	# Get the worst case direction vector (normalized)
	f_dir = computefi(alphaidx, fi_indices, H; normalize=true)
	
	# Apply fault: f = mag * direction
	fault_vector_px = ø * f_dir
	
	# Inject fault into observations
	faulty_observations = noisy_observations .+ [
		ProjectionPoint(fault_vector_px[i], fault_vector_px[i+1])
		for i in 1:2:length(fault_vector_px)
	]
	cam_pose_est_faulty = estimatepose6dof(
	    PointFeatures(runway_corners, faulty_observations)
	)
	cam_pose_est_faulty, faulty_observations
end

# ╔═╡ 6e9e1c89-01f2-447d-8335-ad26881a6773
function integrity_root_objective(ø_nounits, (; alphaidx, fi_indices, H, noisy_observations))
	cam_pose_est_faulty, faulty_observations = estimate_pose_with_faultmagnitude(ø_nounits*px, (; alphaidx, fi_indices, H, noisy_observations))
	# Run Integrity Monitor
	# We only care about the p_value here
	result = compute_integrity_statistic(
		cam_pose_est_faulty[(:pos, :rot)]..., # linearized around current est is fine for H checks
		runway_corners,
		faulty_observations,
		2.0*I(length(runway_corners)*2)
	)
	return result.p_value - 0.05
end


# ╔═╡ df5a6bf7-3f5b-4804-99de-291bdabeacdb
lines(-100:1:100, ø->integrity_root_objective(ø, (; alphaidx=3, fi_indices=[1, 2], H, noisy_observations)))

# ╔═╡ 5aaf7a5a-6cf9-4a1e-9825-e8fc2d6e3d75
begin
	# Define the function for the line search: p_value - 0.05 = 0
	# We search for the fault magnitude 'mag' that puts us exactly on the detection boundary.
	
	# Perform Line Search
	# We look for a root between 0 and 100 pixels of fault magnitude
	tspan = (0.0, 100.0)
	ps = (; alphaidx, fi_indices, H, noisy_observations)
	prob = IntervalNonlinearProblem(integrity_root_objective, tspan, ps)
	sol = solve(prob)
	@assert BracketingNonlinearSolve.SciMLBase.successful_retcode(sol)
	ø_at_boundary = sol.u*px
	
	# Calculate the actual Position Error at this boundary magnitude
	# Error = Slope * Magnitude * Direction_Projection? 
	# Easier to just re-calculate the bias using the linear model: s₀ᵀ f
	
	# Re-compute f vector at the boundary magnitude
	cam_pose_est_faulty, _ = estimate_pose_with_faultmagnitude(ø_at_boundary, ps)
	experimental_max_error = (cam_pose_est_faulty.pos - cam_pos_est)[alphaidx]

	md"""
	### Experimental Verification
	Found the detection boundary via line search:
	
	*   **Fault Magnitude at Boundary:** $(round(ø_at_boundary, digits=2)) px
	*   **Worst Case Error (Experiment):** $(round(typeof(1.0m), experimental_max_error, digits=4))
	
	*Diff: $(round(typeof(1.0m), abs(analytic_max_error - experimental_max_error), digits=5))*
	"""
end

# ╔═╡ a5214c33-0e64-4ac2-8484-ca03bbf660ad
let
	#sigma_val = sqrt(2)
	# --- Helper Functions for 3-DOF Experiment ---

	# 1. Custom 3-DOF Integrity Statistic
	#    Calculates q^2 (SSE) for a fixed-rotation scenario
	function compute_integrity_statistic_3dof(pos, rot, corners, obs, cov_scalar=2.0)
		# A. Compute 3-column Jacobian H (Position only)
		H_full = RunwayLib.compute_H(pos, rot, corners)
		H3 = H_full[:, 1:3] # Take first 3 cols (px/m)
		
		# B. Compute Parity Matrix P = I - H(H'H)^-1 H'
		#    (Using raw Float64 arithmetic for robust projection)
		P_3dof = I - H3 * pinv(H3)
		
		# C. Compute Whitened Residuals
		#    q^2 = r_w' * P * r_w
		sigma = sqrt(cov_scalar)
		
		preds = [project(pos, rot, c) for c in corners]
		# Residuals in pixels (Magnitude)
		r_vec = reduce(vcat, [obs[i] - preds[i] for i in 1:length(obs)]) .|> _ustrip(px)
		
		r_whitened = r_vec ./ sigma
		sse = r_whitened' * P_3dof * r_whitened
		
		# D. P-value (DOF = n_meas - 3)
		dof = length(obs)*2 - 3
		p_val = 1 - cdf(Chisq(dof), sse)
		return (; sse, p_value=p_val, dof)
	end

	# 2. Custom 3-DOF Analytic Slope Calculation
	#    Uses UnitfulMatrix to handle units correctly (consistent with notebook)
	function compute_slope_3dof(alphaidx, fi_indices, H_full_ref)
		# Attach units: Position cols are px/m
		H3 = H_full_ref[:, 1:3] * px/m |> UnitfulMatrix
		
		# Extraction Vector s0
		alpha = zeros(3); alpha[alphaidx] = 1
		S0 = pinv(H3)
		s0 = S0' * alpha
		
		# Parity Matrix P
		P = I - H3 * S0
		
		# Fault Matrix A
		n = size(H3, 1)
		nf = length(fi_indices)
		A = sparse(collect(fi_indices), 1:nf, ones(nf), n, nf) |> Matrix
		
		# Slope Squared (g^2)
		m_vec = A' * s0
		vis_mat = A' * P * A
		g2 = m_vec' * (vis_mat \ m_vec)
		return sqrt(g2)
	end

	# 3. Optimization Objective for Line Search
	function integrity_objective_3dof(ø_val, params)
		(; alphaidx, fi_indices, H_ref_3, obs_ref, rot_fixed, corners) = params
		
		# A. Construct Fault Vector
		#    (Use 3-DOF H for worst-case direction)
		f_dir = computefi(alphaidx, fi_indices, H_ref_3; normalize=true, sixdof=false)
		f_vec = ø_val * f_dir * px
		
		# B. Inject Fault
		obs_faulty = obs_ref .+ [ProjectionPoint(f_vec[i], f_vec[i+1]) for i in 1:2:length(f_vec)]
		
		# C. Estimate Position (3-DOF)
		#    (Rotation is locked to rot_fixed)
		pos_new = estimatepose3dof(PointFeatures(corners, obs_faulty), NO_LINES, rot_fixed).pos
		
		# D. Check Integrity
		res = compute_integrity_statistic_3dof(pos_new, rot_fixed, corners, obs_faulty, 2.0)
		
		return res.p_value - 0.05
	end

	# --- Run Experiment ---
	
	# Setup: Select Crosstrack (2) to verify linearity match
	target_alphaidx = 1
	target_fi_indices = [1, 2] 
	
	# Get Jacobian at nominal estimate
	H_6dof = RunwayLib.compute_H(cam_pos_est, cam_rot_est, runway_corners)
	H_3dof_ref = H_6dof[:, 1:3]
	
	# 1. Calculate Analytic Max Error
	slope_3 = compute_slope_3dof(target_alphaidx, target_fi_indices, H_6dof)
	dof_3 = length(runway_corners)*2 - 3
	T_chisq_3 = quantile(Chisq(dof_3), 0.95)
	sigma_val = sqrt(2.0)*px 
	
	analytic_max_error_3 = slope_3 * sigma_val * sqrt(T_chisq_3)
	
	# 2. Calculate Experimental Max Error
	search_params = (; 
		alphaidx=target_alphaidx, 
		fi_indices=target_fi_indices, 
		H_ref_3=H_3dof_ref, 
		obs_ref=noisy_observations, 
		rot_fixed=cam_rot_est, 
		corners=runway_corners
	)
	
	# Line search for detection boundary
	prob_3 = IntervalNonlinearProblem(integrity_objective_3dof, (0.0, 200.0), search_params)
	sol_3 = solve(prob_3)
	mag_boundary_3 = sol_3.u
	
	# Measure error at boundary
	f_dir_3 = computefi(target_alphaidx, target_fi_indices, H_3dof_ref; normalize=true, sixdof=false)
	f_vec_final = mag_boundary_3 * f_dir_3 * px
	obs_final_3 = noisy_observations .+ [ProjectionPoint(f_vec_final[i], f_vec_final[i+1]) for i in 1:2:length(f_vec_final)]
	
	pos_final_3 = estimatepose3dof(PointFeatures(runway_corners, obs_final_3), NO_LINES, cam_rot_est).pos
	
	experimental_error_3 = abs((pos_final_3 - cam_pos_est)[target_alphaidx])

	# --- Output Results ---
	md"""
	### 3-DOF Verification (Locked Rotation)
	
	This experiment isolates the position states. We expect **Cross-track (2)** to match perfectly (Linear), 
	while **Along-track (1)** maintains a discrepancy (Perspective Nonlinearity).
	
	*   **Axis:** $(target_alphaidx == 1 ? "Along-track" : "Cross-track")
	*   **Analytic Max Error:** $(round(typeof(1.0m), analytic_max_error_3, digits=4))
	*   **Experimental Max Error:** $(round(typeof(1.0m), experimental_error_3, digits=4))
	*   **Difference:** $(round(typeof(1.0m), abs(analytic_max_error_3 - experimental_error_3), digits=4))
	"""
end


# ╔═╡ Cell order:
# ╠═46af6473-88bf-49b9-8dc9-0a72e995f784
# ╠═b5b8f3c8-c4dc-11f0-82e6-e3e1218a8fd8
# ╟─64d2c0fd-2542-4b2c-80f6-134ed8434c3b
# ╠═47423636-18d6-42cb-85e6-4a0909dc168d
# ╠═ddee502b-6245-45eb-b4cc-ce4a4f749fcf
# ╠═2fe79916-81bf-4487-bd21-4656783cc4c6
# ╠═020be658-c6dc-48a3-abb8-34cf8b1fd449
# ╠═951488bb-bf5a-434e-8251-8664ca58ee7d
# ╠═3fae75fa-7760-4878-b97f-888ed1dbbf0f
# ╠═b377ee57-1d61-4787-bb9d-ed2b760ef23d
# ╠═c6a57e0f-50c7-461a-a6e8-9281991b9e44
# ╠═b027b7ad-098e-4048-97d1-f4ce311c5ac4
# ╠═128eddd5-1607-4188-b824-747c34ad5572
# ╠═36e1df5f-9cf1-43d1-a2a4-9e63c56ae7c8
# ╠═49755fa4-babc-43a7-86db-2afcc96b18f7
# ╠═56e87bb5-b7e6-45c1-a2cb-95e0aaf2a000
# ╠═59a0ab1e-0360-4d24-9320-fb3966062b9d
# ╠═120a3051-4909-4e65-a35d-82e76b706567
# ╠═b495605a-ffe7-4783-a490-1d635731da0a
# ╠═a530d3e4-22db-4744-8b39-5947be16772c
# ╠═683af497-4a84-40dd-87e8-0e06ba04d6a3
# ╠═6e3438b3-81b1-4055-87f9-28731bd674c2
# ╠═532e4b44-f7f6-47b6-bb49-58257d2b1157
# ╠═fe706472-f0f1-49cd-94db-ca4537012491
# ╠═a76ba211-2106-40b2-a326-cac15b4545c4
# ╠═e00d78a4-0d5c-4e63-b0de-e2a5cac79abd
# ╠═05fd4f82-3e37-4f50-83ed-66474a8f3003
# ╠═41d7ef29-0a65-488e-9a28-1d625c61703c
# ╠═c208ce81-b980-405e-8698-f632e1898630
# ╠═f80aee26-b1d2-4183-aec8-2187f9c2cdfe
# ╠═ef5eb470-af13-4491-a5a4-d634e41bf6f6
# ╠═58a21d5e-8a66-45b2-8202-aee966463df3
# ╠═6e9e1c89-01f2-447d-8335-ad26881a6773
# ╠═df5a6bf7-3f5b-4804-99de-291bdabeacdb
# ╠═5aaf7a5a-6cf9-4a1e-9825-e8fc2d6e3d75
# ╠═a5214c33-0e64-4ac2-8484-ca03bbf660ad
