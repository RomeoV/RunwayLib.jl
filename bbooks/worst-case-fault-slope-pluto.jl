### A Pluto.jl notebook ###
# v0.20.20

using Markdown
using InteractiveUtils

# ╔═╡ b5b8f3c8-c4dc-11f0-82e6-e3e1218a8fd8
import Pkg; Pkg.activate(".")

# ╔═╡ 47423636-18d6-42cb-85e6-4a0909dc168d
begin
	using Revise
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
	using PlutoUI
	using IntervalSets
	using RunwayLib.StaticArrays
	import RunwayLib: compute_worst_case_fault_direction_and_slope
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

# ╔═╡ c3018bf9-35fc-4234-bb52-53a627c2accf
PlutoUI.TableOfContents()

# ╔═╡ fe109e18-f485-40e8-8af6-034fb0990463
md"# Setup Problem"

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
px_std = sqrt(2.0)px  # this corresponds to the default noise model which has px_var = 4

# ╔═╡ e2debdf2-3c03-473f-9f09-8bb9a145e245
begin
	noise_level = sqrt(2.0)
	sigmas = noise_level * ones(length(runway_corners))
	noise_cov = Diagonal(repeat(sigmas .^ 2, inner=2))
end

# ╔═╡ b377ee57-1d61-4787-bb9d-ed2b760ef23d
noisy_observations = let; Random.seed!(1)
	[p + ProjectionPoint(px_std*randn(2)) for p in true_observations]
end

# ╔═╡ c6a57e0f-50c7-461a-a6e8-9281991b9e44
(cam_pos_est, cam_rot_est) = estimatepose6dof(
    PointFeatures(runway_corners, noisy_observations)
)[(:pos, :rot)]

# ╔═╡ b027b7ad-098e-4048-97d1-f4ce311c5ac4
H0 = RunwayLib.compute_H(cam_pos, cam_rot, runway_corners)

# ╔═╡ 36e1df5f-9cf1-43d1-a2a4-9e63c56ae7c8
cycle(xs::AbstractVector) = xs[[eachindex(xs); first(eachindex(xs))]]

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

# ╔═╡ 58a21d5e-8a66-45b2-8202-aee966463df3
function estimate_pose_with_faultmagnitude(ø, (; alphaidx, fi_indices, H, noisy_observations))
	# Get the worst case direction vector (normalized)
	f_dir = RunwayLib.compute_worst_case_fault_direction_and_slope(alphaidx, fi_indices, H, noise_cov)[1]
	
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
		noise_cov #2.0*I(length(runway_corners)*2)
	)
	return result.p_value - 0.05
end


# ╔═╡ 09c40df5-f975-4d86-ae81-4a35a67f647b
md"# Analytic vs Experimental Validation"

# ╔═╡ df5a6bf7-3f5b-4804-99de-291bdabeacdb
lines(-100:1:100, ø->integrity_root_objective(ø, (; alphaidx, fi_indices, H=H0, noisy_observations)) ; axis=(; xlabel="slider", ylabel="p value"))

# ╔═╡ 18ebe84e-5710-48b7-9849-130b5b55715c
function get_analytic_max_error(alphaidx, fi_indices, H, px_std)
	@show alphaidx " FROM ANALYTIC"
	
	slope_g = RunwayLib.compute_worst_case_fault_direction_and_slope(alphaidx, fi_indices, H, noise_cov)[2] * m
	g_wo_noise = RunwayLib.compute_worst_case_fault_direction_and_slope_wo_noise(alphaidx, fi_indices, H)[2] * m / px
	
	# 2. Determine the Detection Threshold (T)
	# The monitor checks if SSE < T². 
	# We need the T corresponding to our p-value requirement (alpha=0.05).
	dof = size(H, 1) - size(H, 2) # Degrees of Freedom = Measurements - States
	T_chisq = quantile(Chisq(dof), 0.95)  # [px^2 / std_px^2]
	
	# 3. Calculate Analytic Max Error
	# Max Error = Slope * Sigma * sqrt(Threshold)
	# Sigma = sqrt(4.0) because we passed 4.0*I as covariance
	sigma_val = px_std
	@assert isapprox(slope_g * sqrt(T_chisq), g_wo_noise * sigma_val * sqrt(T_chisq); rtol=1e-4)
	analytic_max_error = slope_g * sqrt(T_chisq)
end

# ╔═╡ 982214a6-fd01-4166-aefe-36f051067be2
md"""
## Get experimental error function
"""

# ╔═╡ 8a4de21a-acbc-4ce0-9315-6cc86376e3b1
function get_experimental_max_error(alphaidx, fi_indices, H, noisy_observations, cam_pose_est)
	@show alphaidx
	@show sum(H)
	experimental_max_error_tpl = map([
				(0.0, 100.0),
				(-100.0, 0.0)
			]) do tspan
		ps = (; alphaidx, fi_indices, H, noisy_observations)
		prob = IntervalNonlinearProblem(integrity_root_objective, tspan, ps)
		sol = solve(prob)
		@assert BracketingNonlinearSolve.SciMLBase.successful_retcode(sol)
		ø_at_boundary = sol.u*px
		
		# Calculate the actual Position Error at this boundary magnitude
		cam_pose_est_faulty, _ = estimate_pose_with_faultmagnitude(ø_at_boundary, ps)
		experimental_max_error = if alphaidx <= 3 
			(cam_pose_est_faulty.pos - cam_pose_est.pos)[alphaidx]
		else 
			reverse(params(cam_pose_est_faulty.rot) - params(cam_pose_est.rot))[alphaidx - 3]
		end
		experimental_max_error
	end |> sort
end

# ╔═╡ a5214c33-0e64-4ac2-8484-ca03bbf660ad
let
	#sigma_val = sqrt(2)
	# --- Helper Functions for 3-DOF Experiment ---

	# 1. Custom 3-DOF Integrity Statistic
	#    Calculates q^2 (SSE) for a fixed-rotation scenario
	function compute_integrity_statistic_3dof(pos, rot, corners, obs, cov_scalar=4.0)
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

	# 3. Optimization Objective for Line Search
	function integrity_objective_3dof(ø_val, params)
		(; alphaidx, fi_indices, H_ref_3, obs_ref, rot_fixed, corners) = params
		
		# A. Construct Fault Vector
		#    (Use 3-DOF H for worst-case direction)
		f_dir = RunwayLib.compute_worst_case_fault_direction_and_slope(alphaidx, fi_indices, H_ref_3, noise_cov)[1]
		f_wo_noise = RunwayLib.compute_worst_case_fault_direction_and_slope_wo_noise(alphaidx, fi_indices, H_ref_3)[1]
		@assert isapprox(f_dir, f_wo_noise; rtol=1e-4)

		f_vec = ø_val * f_dir * px
		
		# B. Inject Fault
		obs_faulty = obs_ref .+ [ProjectionPoint(f_vec[i], f_vec[i+1]) for i in 1:2:length(f_vec)]
		
		# C. Estimate Position (3-DOF)
		#    (Rotation is locked to rot_fixed)
		pos_new = estimatepose3dof(PointFeatures(corners, obs_faulty), NO_LINES, rot_fixed).pos
		
		# D. Check Integrity
		res = compute_integrity_statistic_3dof(pos_new, rot_fixed, corners, obs_faulty, 4.0)
		
		return res.p_value - 0.05
	end

	# --- Run Experiment ---
	
	# Setup: Select Crosstrack (2) to verify linearity match
	target_alphaidx = 3
	target_fi_indices = [1, 2, 3, 4, 5] 
	
	# Get Jacobian at nominal estimate
	H_6dof = RunwayLib.compute_H(cam_pos_est, cam_rot_est, runway_corners)
	H_3dof_ref = H_6dof[:, 1:3]
	
	slope_3 = RunwayLib.compute_worst_case_fault_direction_and_slope(target_alphaidx, target_fi_indices, H_3dof_ref, noise_cov)[2] * m
	slope_3_wo_noise = RunwayLib.compute_worst_case_fault_direction_and_slope_wo_noise(target_alphaidx, target_fi_indices, H_3dof_ref)[2] * m / px

	dof_3 = length(runway_corners)*2 - 3
	T_chisq_3 = quantile(Chisq(dof_3), 0.95)
	old_sigma_val = sqrt(2.0)*px
	
	analytic_max_error_3 = slope_3 * sqrt(T_chisq_3)
	new_sigma_val = 2.0*px
	@assert isapprox(analytic_max_error_3, slope_3_wo_noise * px_std * sqrt(T_chisq_3); rtol=1e-4)
	@show slope_3_wo_noise * new_sigma_val * sqrt(T_chisq_3), slope_3_wo_noise * old_sigma_val * sqrt(T_chisq_3)
	
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
	f_dir_3 = RunwayLib.compute_worst_case_fault_direction_and_slope(target_alphaidx, target_fi_indices, H_3dof_ref, noise_cov)[1]
	f_wo_noise_3 = RunwayLib.compute_worst_case_fault_direction_and_slope_wo_noise(target_alphaidx, target_fi_indices, H_3dof_ref)[1]
	@assert isapprox(f_dir_3, f_wo_noise_3; rtol=1e-4)
	
	f_vec_final = mag_boundary_3 * f_dir_3 * px
	obs_final_3 = noisy_observations .+ [ProjectionPoint(f_vec_final[i], f_vec_final[i+1]) for i in 1:2:length(f_vec_final)]
	
	pos_final_3 = estimatepose3dof(PointFeatures(runway_corners, obs_final_3), NO_LINES, cam_rot_est).pos
	
	experimental_error_3 = abs((pos_final_3 - cam_pos_est)[target_alphaidx])

	# --- Output Results ---
	md"""
	## 3-DOF Verification (Locked Rotation)
	
	This experiment isolates the position states. We expect **Cross-track (2)** to match perfectly (Linear), 
	while **Along-track (1)** maintains a discrepancy (Perspective Nonlinearity).
	
	*   **Axis:** $(target_alphaidx == 1 ? "Along-track" : "Cross-track")
	*   **Analytic Max Error:** $(round(typeof(1.0m), analytic_max_error_3, digits=4))
	*   **Experimental Max Error:** $(round(typeof(1.0m), experimental_error_3, digits=4))
	*   **Difference:** $(round(typeof(1.0m), abs(analytic_max_error_3 - experimental_error_3), digits=4))
	"""
end


# ╔═╡ 5c6760d3-7aed-4a17-a5e6-5dbf420fc6e1
	md"# Refactored visualization"

# ╔═╡ 76ad9899-7dd2-4795-b1b6-b44e34b747af
md"""
Integrity check threshold is set such that we tolerate 5% false alarm rate.
"""

# ╔═╡ 25348c41-f099-459c-9b19-250a66a01cab
begin
function setup_ui(fig, _ctx)
    # Slider Grid
    slg = Makie.SliderGrid(
        fig[3, 1:2],
        (label="perturbation dir 1: ", range=-100:0.5:100, startvalue=0.0),
        (label="perturbation dir 2:", range=-100:0.5:100, startvalue=0.0),
        (label="random pertubation: ", range=-100:0.5:100, startvalue=0.0),
        (label="worst case pertubation: ", range=-10:0.05:10, startvalue=0.0),
    )
    sl1, sl2, sl3, sl4 = slg.sliders

    # Control Layout Area
    control_layout = GridLayout(fig[2, 1:2])

    # Corner Selections
    (; gl, cbs, axismenu) = setup_corner_selections(control_layout[1, 1])
    
    # Visual Toggle
    visual_menu = GridLayout(control_layout[1, 2]; tell_width=false)
    Label(visual_menu[1, 1], text="Show Runway", halign=:right)
    show_runway = Toggle(visual_menu[1, 2], active=false)
    Label(visual_menu[2, 1], text="Estimate Rotation")
    do_estimate_rot = Toggle(visual_menu[2, 2], active=true)

    return (; sl1, sl2, sl3, sl4, cbs, axismenu, show_runway, do_estimate_rot, control_layout)
end

# 2. Computations (The Pipeline)
function do_computations(ui, ctx)
	(; px_std, true_observations, noisy_observations, runway_corners) = ctx
    (; sl1, sl2, sl3, sl4, cbs, axismenu, do_estimate_rot) = ui
    cb1, cb2, cb3, cb4 = cbs


    # First we do the regulary noisy pose estimation, either 3 or 6 dof
    cam_pose_est_noisy = @lift if $(ui.do_estimate_rot.active)
        estimatepose6dof(
            PointFeatures(ctx.runway_corners, ctx.noisy_observations)
        )
    else
        estimatepose3dof(
            PointFeatures(ctx.runway_corners, ctx.noisy_observations),
            NO_LINES,
            ctx.cam_rot
        )
    end
    cam_pos_est_noisy = @lift $(cam_pose_est_noisy).pos
    cam_rot_est_noisy = @lift $(cam_pose_est_noisy).rot

    H = @lift RunwayLib.compute_H($cam_pos_est_noisy, $cam_rot_est_noisy, ctx.runway_corners)
	Q = @lift let
		Qt = nullspace($(H)')
		Qt'
	end

    # Helper Logic
    fi_indices = @lift let
        idx = findall([$(cb1.checked), $(cb2.checked), $(cb3.checked), $(cb4.checked)])
        sort([(1:2:8)[idx]; (2:2:8)[idx]])
    end

    alphaidx = @lift(Dict(1=>1, 2=>2, 3=>3, 4=>6, 5=>5, 6=>4)[$(axismenu.i_selected)])
    
    # Points Calculations
    yobs_pts = Point2.(true_observations)
    yperturb = @lift $(Q)' * [$(sl1.value); $(sl2.value)]
    yperturb_pts = @lift ProjectionPoint.(eachcol(reshape($yperturb, 2, :))) * px
	drand = normalize(randn(Random.MersenneTwister(1), 8))
    yrand_pts = @lift ProjectionPoint.(eachcol(reshape($(sl3.value) * drand, 2, :))) * px
    f_i = @lift RunwayLib.compute_worst_case_fault_direction_and_slope($alphaidx, $fi_indices, $H, noise_cov)[1]
	
    yfi_pts = @lift let sl4val = sign($(sl4.value))*exp(abs($(sl4.value)))
        # we have an exponential "gain" on this slider
        ProjectionPoint.(eachcol(reshape( sl4val * $f_i, 2, :))) * px
    end

    # Summation and Pose Estimation
    perturbed_observations = @lift ctx.noisy_observations .+ $yperturb_pts .+ $yrand_pts .+ $yfi_pts
    
    cam_pose_est_pert = @lift if $(ui.do_estimate_rot.active)
        estimatepose6dof(
            PointFeatures(ctx.runway_corners, $perturbed_observations)
        )
    else
        estimatepose3dof(
            PointFeatures(ctx.runway_corners, $perturbed_observations),
            NO_LINES,
            ctx.cam_rot
        )
    end
    cam_pos_est_pert = @lift $(cam_pose_est_pert).pos
    cam_rot_est_pert = @lift $(cam_pose_est_pert).rot

    on(ui.do_estimate_rot.active) do _
		println(cam_rot_est_noisy - cam_rot_est_pert)
    end

    # Integrity Check
    passed = @lift compute_integrity_statistic(
        $(cam_pos_est_pert), $(cam_rot_est_pert),
        ctx.runway_corners,
        $(perturbed_observations), 
        noise_cov
    ).p_value > 0.05

	@info "HELLO"
	on(alphaidx) do alphaidx
		@info "WORLD"
		@info "From the viz: $(alphaidx)"
	end
	analytic_worst_case = @lift get_analytic_max_error($alphaidx, $fi_indices, $H, ctx.px_std)
	experimental_worst_case = @lift get_experimental_max_error($alphaidx, $fi_indices, $H, ctx.noisy_observations, $cam_pose_est_noisy)

    return (; yobs_pts, yperturb_pts, yrand_pts, yfi_pts, 
              perturbed_observations, cam_pose_est_pert, 
              cam_pos_est_noisy, cam_rot_est_noisy,
              cam_pos_est_pert, cam_rot_est_pert, passed, analytic_worst_case, experimental_worst_case)
end

# 3. Visualization
function setup_plots(fig, ui, data, ctx)
	(; aircraft_model, runway_corners, true_observations) = ctx
    colors = Makie.wong_colors()
    c1, c4, c7 = colors[1], colors[4], colors[7]
    
    # --- Text Labels (Delta) ---
    # Note: Layout is inserted into the control_layout passed from UI
    ## POSE
    pose_delta_layout = GridLayout(ui.control_layout[1, 0])
    Label(pose_delta_layout[0, 0:1], text="Pose Delta", font=:bold)
    Label(pose_delta_layout[1, 0], text="x = \ny = \nz = ")
    
    Label(pose_delta_layout[1, 1], text=@lift(let 
        diff = ($(data.cam_pos_est_pert) - cam_pos_est)
        s = repr("text/plain", round.([typeof(1.0m)], diff; digits=2))
        split(s, '\n')[2:end] |> x->join(x, '\n')
    end), halign=:right)
    
    rowgap!(pose_delta_layout, 0); colgap!(pose_delta_layout, 0)
    
    ## ATTITUDE
    Label(pose_delta_layout[0, 2:4], text="Attitude Delta", font=:bold)
    Label(pose_delta_layout[1, 2], text="roll\npitch\nyaw", justification=:left)
    Label(pose_delta_layout[1, 3], text="=\n=\n=", justification=:left)
    
    Label(pose_delta_layout[1, 4], text=@lift(let 
        diff = params($(data.cam_rot_est_pert)) - params($(data.cam_rot_est_noisy))
        s = repr("text/plain", round.([typeof(1.0°)], reverse(rad2deg.(diff.*rad)); digits=1))
        split(s, '\n')[2:end] |> x->join(x, '\n')
    end), halign=:right)

	## Worst case
	worst_case_layout = GridLayout(pose_delta_layout[0:1, 5])
	Label(worst_case_layout[1,1], text="Analytic Worst Case", font=:bold, halign=:left)
	Label(worst_case_layout[2,1], text=@lift(let
		worst_case_rnd = round(typeof(1.0m), $(data.analytic_worst_case); sigdigits=2)
		string(0m±worst_case_rnd)
	end), valign=:top, halign=:left)
	Label(worst_case_layout[3,1], text="Line Search Worst Case", font=:bold, halign=:left)
	Label(worst_case_layout[4,1], text=@lift(let
		worst_case_tpl = round.(typeof(1.0m), $(data.experimental_worst_case); sigdigits=2)
		string(worst_case_tpl[1] .. worst_case_tpl[2])
	end), valign=:top, halign=:left)
	rowgap!(worst_case_layout, 0)

    # --- 3D Pose View ---
    ax3 = Axis3(fig[1, 1]; title="Pose Estimate", aspect=:data,
                xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
    
    meshargs = (; marker=aircraft_model, markersize=@lift($(ui.show_runway.active) ? 3 : 1/3))
    
    meshscatter!(ax3, @lift([$(data.cam_pos_est_noisy) .|> _ustrip(m)]);
                 color=c1, label="Reference", rotation=@lift(to_corrected_quat($(data.cam_rot_est_noisy))),
                 meshargs...)
                 
    meshscatter!(ax3, @lift([$(data.cam_pos_est_pert) .|> _ustrip(m)]);
                 color=@lift(($(data.passed) ? :green : :red, 1.0)),
                 label="Perturbed", rotation=@lift(to_corrected_quat($(data.cam_rot_est_pert))),
                 meshargs...)
                 
    axislegend(ax3)
    poly!(ax3, [pt .|> _ustrip(m) for pt in cycle(runway_corners)], visible=ui.show_runway.active)

    # Trigger limit resets
    on(_ -> reset_limits!(ax3), data.cam_pos_est_pert)
    on(_ -> reset_limits!(ax3), ui.show_runway.active)

    # --- 2D Projection View ---
    ax = Axis(fig[1, 2]; yreversed=true, title="Perturbed Observations", aspect=DataAspect())
    
    scatterlines!(ax, [obs .|> _ustrip(px) for obs in cycle(true_observations)])
    
    # Dashed total perturbation
    scatterlines!(ax, @lift([obs .|> _ustrip(px) for obs in cycle(
        data.yobs_pts .+ $(data.yperturb_pts) .+ $(data.yrand_pts) .+ $(data.yfi_pts))]);
        color=(:yellow, 0.5), linestyle=:dash)

    # Incremental arrows
    arrows2d!(ax, [obs .|> _ustrip(px) for obs in data.yobs_pts], 
              data.yperturb_pts; color=:red)
              
    arrows2d!(ax, @lift([obs .|> _ustrip(px) for obs in (data.yobs_pts .+ $(data.yperturb_pts))]),
              data.yrand_pts; color=c7)
              
    arrows2d!(ax, @lift([obs .|> _ustrip(px) for obs in (data.yobs_pts .+ $(data.yperturb_pts) .+ $(data.yrand_pts))]),
              data.yfi_pts; color=c4)

    on(_ -> reset_limits!(ax), data.yperturb_pts)
    on(_ -> reset_limits!(ax), data.yrand_pts)
    on(_ -> reset_limits!(ax), data.yfi_pts)
end
context = (; true_observations, noisy_observations, runway_corners, cam_pos, cam_rot, aircraft_model, px_std)
# 4. Main Orchestrator
with_theme(theme_black()) do
	fig = Figure(; size=(1200, 600))
	
	ui = setup_ui(fig, context)
	data = do_computations(ui, context)
	setup_plots(fig, ui, data, context)
	
	fig
end
end

# ╔═╡ Cell order:
# ╠═46af6473-88bf-49b9-8dc9-0a72e995f784
# ╠═b5b8f3c8-c4dc-11f0-82e6-e3e1218a8fd8
# ╟─64d2c0fd-2542-4b2c-80f6-134ed8434c3b
# ╠═47423636-18d6-42cb-85e6-4a0909dc168d
# ╠═c3018bf9-35fc-4234-bb52-53a627c2accf
# ╠═fe109e18-f485-40e8-8af6-034fb0990463
# ╠═ddee502b-6245-45eb-b4cc-ce4a4f749fcf
# ╠═2fe79916-81bf-4487-bd21-4656783cc4c6
# ╠═020be658-c6dc-48a3-abb8-34cf8b1fd449
# ╠═951488bb-bf5a-434e-8251-8664ca58ee7d
# ╠═3fae75fa-7760-4878-b97f-888ed1dbbf0f
# ╠═e2debdf2-3c03-473f-9f09-8bb9a145e245
# ╠═b377ee57-1d61-4787-bb9d-ed2b760ef23d
# ╠═c6a57e0f-50c7-461a-a6e8-9281991b9e44
# ╠═b027b7ad-098e-4048-97d1-f4ce311c5ac4
# ╠═36e1df5f-9cf1-43d1-a2a4-9e63c56ae7c8
# ╠═59a0ab1e-0360-4d24-9320-fb3966062b9d
# ╠═120a3051-4909-4e65-a35d-82e76b706567
# ╠═b495605a-ffe7-4783-a490-1d635731da0a
# ╠═58a21d5e-8a66-45b2-8202-aee966463df3
# ╠═6e9e1c89-01f2-447d-8335-ad26881a6773
# ╟─09c40df5-f975-4d86-ae81-4a35a67f647b
# ╠═df5a6bf7-3f5b-4804-99de-291bdabeacdb
# ╠═18ebe84e-5710-48b7-9849-130b5b55715c
# ╟─982214a6-fd01-4166-aefe-36f051067be2
# ╠═8a4de21a-acbc-4ce0-9315-6cc86376e3b1
# ╠═a5214c33-0e64-4ac2-8484-ca03bbf660ad
# ╟─5c6760d3-7aed-4a17-a5e6-5dbf420fc6e1
# ╟─76ad9899-7dd2-4795-b1b6-b44e34b747af
# ╠═25348c41-f099-459c-9b19-250a66a01cab
