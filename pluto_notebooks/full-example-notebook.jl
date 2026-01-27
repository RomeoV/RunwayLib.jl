### A Pluto.jl notebook ###
# v0.20.20

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ b4fee6f9-7344-4d8c-b16e-33eff1cede68
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
	import RunwayLib.ProbabilisticParameterEstimators
	import RunwayLib: UncorrGaussianNoiseModel
end

# ╔═╡ f6bfcc72-21f3-4de4-b50f-b044b20d6e76
md"""
# RunwayLib.jl: Runway Pose Estimation
Stanford Intelligent Systems Laboratory (SISL), Stanford University

**Abstract**

> This notebook demonstrates how to:
> 1. **Estimate aircraft pose** under uncertainty using point and line features.
> 2. **Calculate an integrity score** and analyze its statistical properties, such as maintaining a 5% false alarm rate.
> 3. **Compute integrity bounds** to provide safety guarantees for vision-based landing systems.
"""

# ╔═╡ e56ca9f7-9c84-4985-89a9-a6f3d60111aa
md"""
# Notebook set-up
"""

# ╔═╡ 1821382b-4378-4e78-88e9-0cd4a4a699c8
html"""<style>
main {
    max-width: 75%;
}
pluto-editor main {
    align-self: center;
    margin-right: 0;
}
"""

# ╔═╡ 9612a518-b17a-467a-a6c8-6661aff7a1d8
PlutoUI.TableOfContents()

# ╔═╡ 09c592ab-17a5-4829-a646-6edb2cb9541e
md"""
# Problem Set-up
Here, we will define a toy set-up with four known world coordinates for the runway corners, as well as a known camera position and rotation. Then, we will project where the true observations will be on the camera screen given the camera position, rotation, and known world points.

Since our observations of the runway corners come from a computer vision-based system, we don't observe the true location of the runway corners but rather some noisy estimate of them. To simulate this noise, we perturb the true observations of the runway corners in random directions to obtain "noisy observations".
"""

# ╔═╡ 308735e9-c796-4404-9b07-e65b3451f9be
begin
	# Define the location of the runway corners
	runway_corners = [
	    WorldPoint(0.0m, 50m, 0m),     # near left
	    WorldPoint(3000.0m, 50m, 0m),  # far left
	    WorldPoint(3000.0m, -50m, 0m),  # far right
	    WorldPoint(0.0m, -50m, 0m),    # near right
	]
	
	# Define the camera position and rotation
	cam_pos = WorldPoint(-2000.0m, 12m, 150m)
	cam_rot = RotZYX(roll=1.5°, pitch=5°, yaw=0°)

	# Choose the noise level
	noise_level = 2.0
	sigmas = noise_level * ones(length(runway_corners))
	noise_cov = Diagonal(repeat(sigmas .^ 2, inner=2))

	# Get the true observations of the runway corners
	true_observations = [project(cam_pos, cam_rot, p) for p in runway_corners]

	# We don't observe the true location of the runway corners, but 
	# rather some noisy estimate of them. Here, we simulate noise.
	noisy_observations = [p + ProjectionPoint(noise_level*randn(2)px) for p in true_observations]
	
end

# ╔═╡ 58a36b23-8d35-4e51-a833-dbd47914b12c
md"""
# Camera Model

The [RunwayLib.jl](https://romeov.github.io/RunwayLib.jl/dev/) camera model implements a standard [pinhole camera model](https://en.wikipedia.org/wiki/Pinhole_camera_model) without any distortion. It is specifically designed for tasks like runway landing pose estimation.

---

### 1. Reference Frames
The library supports two different projection plane reference frames:

* **`:centered`**: The origin is at the image center; $u$ points left and $v$ points up.
* **`:offset`**: The origin is at the top-left; $u$ points right and $v$ points down.

### 2. Configuration Methods
You can define the camera using two different structures:

| Method | Description |
| :--- | :--- |
| **`CameraConfig`** | Defined by physical properties: focal length, pixel size, and image dimensions. |
| **`CameraMatrix`** | Uses a $3 \times 3$ projection matrix. Currently, this primarily supports the `:offset` frame. |

### 3. Feature Support
* **Points**: Supports 3D world point projection to 2D image coordinates.
* **Lines**: Supports line features for `:offset` models, parameterized by their angle and radius (Hough transform).
"""

# ╔═╡ 1f00d3db-5cbb-4f72-b070-b21aab5325b8
begin
	# Define the offset camera configuration
    focal_length = 25mm
    pixel_size = 5μm/px
    camconf_offset = CameraConfig{:offset}(focal_length, pixel_size, 4096.0px, 2048.0px)
	
	# Project a specific world point (e.g., the first runway corner) to 2D
	# Replacing the undefined 'world_pt' with 'runway_corners[1]'
	project(cam_pos, cam_rot, runway_corners[1], camconf_offset)
end

# ╔═╡ a29d1cf7-e04f-42cb-b83f-aae1de898cdf
md"""
# Line Projections

For **`:offset`** camera models, [RunwayLib.jl](https://romeov.github.io/RunwayLib.jl/dev/camera_model/#Line-Projections) supports line features. Instead of using raw pixel coordinates for endpoints, lines are parameterized using the **Hough transform** (angle and radius).

### Geometric Representation
Lines are defined with respect to the **offset origin** (top-left) and are represented by two parameters:
* **$\theta$ (Angle)**: The angle of the normal vector from the origin to the line.
* **$r$ (Radius)**: The perpendicular distance from the origin to the line.
"""

# ╔═╡ cf5b9e5a-9449-4d03-b65c-8ee34f0cdbd6
begin
	# 1. Define the line segments using your existing runway corners
	line_pts = [
		(runway_corners[1], runway_corners[2]), # Left edge
		(runway_corners[3], runway_corners[4]), # Right edge
	]
	
	# 2. Project and convert the first line as an example
	# Note: Ensure you use camconf_offset for line features
	p1, p2 = line_pts[1]
	proj1 = project(cam_pos, cam_rot, p1, camconf_offset)
	proj2 = project(cam_pos, cam_rot, p2, camconf_offset)
	
	# 3. Represent in Hough space
	line_feature = getline(proj1, proj2)
end

# ╔═╡ f692b3ae-6611-4905-bd74-70779155cc9f
begin
	# Get the true observations of the lines
	true_lines = map(line_pts) do (p1, p2)
		proj1 = project(cam_pos, cam_rot, p1)
		proj2 = project(cam_pos, cam_rot, p2)
		getline(proj1, proj2)
	end
	
	# Just like with the runway corners, we don't observe the true location 
	# of the lines but rather some noisy estimate of them.
	observed_lines = [
	  Line(
		r + 1px*randn(),
		theta + deg2rad(1°)*randn()
	  )
	  for (; r, theta) in true_lines
	]
end

# ╔═╡ 78b0a026-3f8f-4af2-afea-2c451976d7f0
md"""
# Pose Estimation
In our paper on [Probabilistic Parameter Estimators and Calibration Metrics for Pose Estimation from Image Features](https://arxiv.org/abs/2407.16223), we presented three probabilitic parameter estimators to estimate the pose of the aircraft.

We can estimate the position of the camera by solving the PnP (Pose from N points) problem. Here, we estimate both the position and rotation, so we have 6 degrees of freedom (3 for position, 3 for rotation). 
"""

# ╔═╡ afe48040-9b95-42e5-8961-9e8ecdc1662b
md"""
## 1. Pose Estimation using World Points
We can estimate the camera pose using specific points in the world, such as runway corners. Point features provide strong constraints for all six degrees of freedom, though their precision is highly dependent on the pixel-level accuracy of the corner detections.
"""

# ╔═╡ 6c6ccd8d-f5a7-41c4-9e16-0148a1b7462a
begin
	(cam_pos_est, cam_rot_est) = estimatepose6dof(
		PointFeatures(runway_corners, noisy_observations)
	)[(:pos, :rot)]
			
	cam_pos_est
end

# ╔═╡ c3c9e87c-4a30-4e15-a085-02d2330f795d
md"""
## 2. Pose Estimation using Line Features
We can also use line features in our pose estimation. Line features can typically improve our altitude and crosstrack estimations, but usually can't improve our alongtrack estimation much because the line projections are constant along the glidepath. See [Line Projections](https://romeov.github.io/RunwayLib.jl/dev/camera_model/#Line-Projections) for more information on the line parameterization.


"""

# ╔═╡ 60f7b871-f48d-4a95-8f7c-82022a0a4daf
begin
	# Now we can estimate the pose with additional line features
	(cam_pos_lines_est, cam_rot_lines_est) = estimatepose6dof(
	    PointFeatures(runway_corners, noisy_observations),
	    LineFeatures(line_pts, observed_lines)
	)[(:pos, :rot)]
	
	cam_pos_lines_est
end

# ╔═╡ d12b5fc0-7819-444a-b297-402448b9960e
md"""
## 3. Noise Models for Pose Estimation

In pose estimation, we often assume that our observations (pixel coordinates of corners or line parameters) are subject to Gaussian noise. [RunwayLib.jl](https://romeov.github.io/RunwayLib.jl/dev/) provides several ways to model this uncertainty:

* **[UncorrGaussianNoiseModel](https://romeov.github.io/RunwayLib.jl/dev/noise_models/#RunwayLib.UncorrGaussianNoiseModel)**: The simplest model, assuming independent, identical Gaussian noise for all observations.
* **[UncorrProductNoiseModel](https://romeov.github.io/RunwayLib.jl/dev/noise_models/#RunwayLib.UncorrProductNoiseModel)**: Allows for independent but non-identical noise across different features.
* **[CorrGaussianNoiseModel](https://romeov.github.io/RunwayLib.jl/dev/noise_models/#RunwayLib.CorrGaussianNoiseModel)**: A more complex model that accounts for correlations between different observations, which is useful when errors in one feature are likely linked to errors in another.

These models allow the probabilistic estimators to weigh more reliable features more heavily when calculating the aircraft's pose.
"""

# ╔═╡ 95b02000-cb5a-4e5f-8984-61850448ec0f
begin
	# New noise covariance matrix using UncorrGaussianNoiseModel
	uncorrgaussiannoise_cov = UncorrGaussianNoiseModel(
		[Distributions.MvNormal([0, 0], diagm([i, i])) for i in 1:4]
	)

	# Now we can estimate the pose with this type of noise
	(cam_pos_lines_est_noise, cam_rot_lines_est2_noise) = estimatepose6dof(
	    PointFeatures(
			runway_corners,
			noisy_observations,
			RunwayLib.CAMERA_CONFIG_OFFSET, 
			uncorrgaussiannoise_cov
		),
	    LineFeatures(line_pts, observed_lines)
	)[(:pos, :rot)]
	
	cam_pos_lines_est_noise
end

# ╔═╡ abb49dbf-be2b-442c-a6bf-f3038303f6c3
md"""
# Integrity Check

Given our estimate of the camera position and rotation, and our knowledge of the known world points of the runway corners, we can reproject where the runway corners should be observed given this information. 

If the reprojected observations are far from our noisy observations (the "inputs" into our pose estimator), this indicates that our noisy observations might be faulty. 
"""

# ╔═╡ 0c865680-4ed3-4429-824b-1a93e3b30ca9
reprojected_observations = [project(cam_pos_est, cam_rot_est, p) for p in runway_corners]

# ╔═╡ 7af47b79-8e9a-4bcf-9b30-882d5eb317c2
md"""
In our [paper](https://arxiv.org/abs/2508.09732), we formalize this idea by comparing the reprojection error of the estimated pose to the magnitude of the predicted uncertainties. To do this, we adapt an algorithm often used in GNSS called Receiver Autonomous Integrity Monitoring. More details in the [documentation](https://romeov.github.io/RunwayLib.jl/dev/integrity_check/).

The `integrity_statistic` function computes the RAIM-adaptation statistic, the p-value of the null hypothesis, and the degrees of freedom, along with some other information. 
"""

# ╔═╡ c457014a-eacc-4892-a246-a4a1a2f2541b
integrity_statistic = compute_integrity_statistic(
    cam_pos, cam_rot,
    runway_corners,
    noisy_observations,
    noise_cov
)

# ╔═╡ e04102c1-f310-46db-ae08-87bdd66ec11f
md"""
## 1. Artificial Disturbance (Fault Detection)
"""

# ╔═╡ bd2b813e-e7dc-4616-8bcd-b189a00e087b
begin
	# 1. Create the Slider UI
	# We use a range from -60 to 60 pixels.
	# Using Slider instead of Select makes it a draggable bar.
	mag_slider = @bind fault_mag Slider(-60.0:0.5:60.0, default=0.0, show_value=true)
	
	md"""
	### Interactive Fault Injection
	Adjust the slider to inject a horizontal fault (pixels) into the first runway corner:
	$(mag_slider)
	"""
end

# ╔═╡ 449bf3a9-8593-4e06-b6ea-9ecded426aa4
begin
	# 1. Dynamically apply the fault from the slider
	# We use the 'fault_mag' variable bound in the previous cell
	faulty_observations = copy(noisy_observations)
	faulty_observations[1] += ProjectionPoint(fault_mag * px, 0.0px)
	
	# 2. Run the integrity check
	# This re-calculates automatically as you drag the slider
	fault_stat = compute_integrity_statistic(
		cam_pos, cam_rot, runway_corners, faulty_observations, noise_cov
	)
	
	# 3. Display the results reactively
	md"""
	With a **$(fault_mag)px** artificial disturbance, the integrity statistic is **$(round(fault_stat.stat, digits=2))** with a p-value of **$(round(fault_stat.p_value, digits=2))**. 
	
	Status: **$(fault_stat.p_value < 0.05 ? "❌ FAULT DETECTED (Untrustworthy)" : "✅ NOMINAL (Integrity Passed)")**
	"""
end

# ╔═╡ 0c897c1b-0e57-4292-89e1-7ee356bd1fa0
md"""
## 2. Empirical Distribution vs. $\chi^2$ Curve

To verify the integrity monitor, we compare the **integrity statistic** under nominal conditions against its theoretical **Chi-squared ($\chi^2$) distribution**.

* **Empirical Histogram (Blue)**: Data from 1,000 Monte Carlo simulations using the `noise_level` from the problem set-up.
* **Theoretical Curve (Red)**: The expected PDF for the calculated degrees of freedom.

The alignment confirms that our noise covariance is correctly calibrated. A right-shifted histogram would indicate underestimated sensor noise.
"""

# ╔═╡ 26f751af-4fd6-4caa-8534-d2401ffa1fb4
begin
	# 1. Sample nominal statistics
	n_samples = 1000
	empirical_stats = map(1:n_samples) do _
		# Generate nominal noisy observations
		nominal_noisy = [p + ProjectionPoint(noise_level * randn(2)px) for p in true_observations]
		compute_integrity_statistic(
			cam_pos, cam_rot, runway_corners, nominal_noisy, noise_cov
		).stat
	end

	# 2. Plotting (Assumes WGLMakie is active)
	f = Figure()
	ax = Axis(f[1,1], title="Empirical Stats vs. Chi-squared (df=$(fault_stat.dofs))",
		xlabel="Statistic Value", ylabel="Density")
	
	# Histogram of empirical samples
	hist!(ax, empirical_stats, normalization=:pdf, bins=30, label="Empirical")
	
	# Theoretical Chi-sq curve
	x_range = 0:0.1:maximum(empirical_stats)
	lines!(ax, x_range, pdf.(Chisq(fault_stat.dofs), x_range), color=:red, label="Theoretical")
	
	axislegend(ax)
	f
end

# ╔═╡ 3fb311c4-69ce-4ef5-a9f6-88f6f7e7f473
md"""
## 3. False Alarm and False Miss Rates
To calculate these rates, we compare the statistics to a threshold ($T$) derived from our $5\%$ false alarm requirement.
"""

# ╔═╡ 5839d7b4-a97f-484b-bdd6-d8446ebe9114
begin
	# 1. Set threshold for 5% False Alarm Rate (FAR)
	α_target = 0.05
	T_threshold = quantile(Chisq(fault_stat.dofs), 1 - α_target)
	
	# 2. Calculate Empirical FAR
	actual_far = sum(empirical_stats .> T_threshold) / n_samples
	
	# 3. Calculate False Miss Rate (FMR)
	# This requires defining a specific "Missed" fault size (e.g., 10px)
	fault_size = 60.0
	miss_samples = map(1:n_samples) do _
		new_noisy_obs = [p + ProjectionPoint(noise_level * randn(2)px) for p in true_observations]
		new_noisy_obs[1] += ProjectionPoint(fault_size*px, 0.0px)
		compute_integrity_statistic(cam_pos, cam_rot, runway_corners, new_noisy_obs, noise_cov).stat
	end
	actual_fmr = sum(miss_samples .< T_threshold) / n_samples
	
	md"""
	### Integrity Performance
	* **Target False Alarm Rate**: $(α_target * 100)%
	* **Empirical False Alarm Rate**: $(round(actual_far * 100, digits=2))%
	* **False Miss Rate (for $(fault_size)px fault)**: **$(round(actual_fmr * 100, digits=2))%**
	"""
end

# ╔═╡ 25806775-15b4-4a89-852a-4e3f27bf623d
md"""
## 4. Worst-Case Fault Analysis

If the p-value from the integrity statistic fall below a certain threshold, often set to 0.05, then we lack integrity/trust in our pose estimate. However, what can we say about our pose estimate if the integrity check passes?

In the next section, we provide protection levels for the pose estimate for each monitored parameter (e.g., height, altitude, etc.). In other words, can determine the maximum deviation in pose from our estimate that could go undetected by our integrity check.
"""

# ╔═╡ 5befb9d5-53bc-4367-9543-0c62894cea9c
# Compute the Jacobian (relates changes in runway corner position to pose space)
H = RunwayLib.compute_H(cam_pos, cam_rot, runway_corners) # H_pos=H[:, 1:3]         # position-only Jacobian

# ╔═╡ a3b64e53-8b18-44c2-84be-83802065b5b5
md"""
We need to choose the indices of measurements in the fault subset. In other words, we need to decide what specific pixel measurements are unreliable. The full set includes horizontal ("right") and vertical ("up") values for each of our four corners: $[y_{[1, \text{right}]}, y_{[1, \text{up}]}, y_{[2, \text{right}]}, y_{[2, \text{up}]}, y_{[3, \text{right}]}, y_{[3, \text{up}]}, y_{[4, \text{right}]}, y_{[4, \text{up}]}]$, where we map the corners as follows: $y_1$ is the near left corner, and $y_2$, $y_3$, $y_4$ are the far left, far right, and near right corners, respectively.

We can choose a subset of these, with the following indices:
- Index 1 = Near left corner, "right" direction
- Index 2 = Near left corner, "up" direction
- Index 3 = Far left corner, "right" direction
- Index 4 = Far left corner, "up" direction
- Index 5 = Far right corner, "right" direction
- Index 6 = Far right corner, "up" direction
- Index 7 = Near right corner, "right" direction
- Index 8 = Near right corner, "up" direction


For example, if we assume faults in the right direction of the near left corner and the up direction of the near right corner, our fault indices would be [1, 8].


"""

# ╔═╡ 1a11b8c8-7534-4f19-a072-b9eb8c94f095
# assume faults in the right and up direction in the near left corner
fault_indices = [1, 2] 

# ╔═╡ 981ab774-a55b-462d-a8b0-b78e921ce620
md"""
We choose a monitored parameter. We have the following options:
- 1 = along-track position
- 2 = cross-track position
- 3 = altitude
- 4 = yaw
- 5 = pitch
- 6 = roll
"""

# ╔═╡ cd7f4a79-8d5b-4553-80ed-a80b6ed48158
begin
	# 1. Create the Dropdown UI for the monitored parameter
	# The keys are the labels shown in the menu, and the values are the indices
	alpha_menu = @bind alpha_idx Select([
		1 => "along-track",
		2 => "cross-track",
		3 => "altitude",
		4 => "yaw",
		5 => "pitch",
		6 => "roll"
	], default=3)

	md"""
	### Monitored Parameter
	Select the aircraft state to monitor for integrity: $(alpha_menu)
	"""
end

# ╔═╡ 631d3b99-a287-4f4a-9494-3ddd5925a1be
md"""
Now, we are ready to compute the worst case fault direction and slope. The algorithm will find the "worst-case" way to move our corner(s) in the fault subset in pixel space to maximize the error in our monitored parameter alpha_idx (like height). The slope represents the ratio of position error to parity residual, or how much our pose deviates in our monitored parameter direction for each standard deviation moved in the fault direction.

In this example, we can see that the worst case fault direction is moving the near left corner in the right direction and a bit up.
"""

# ╔═╡ 1ab2cf0b-3eae-4958-92d5-84022b733d7b
# Compute worst-case undetected fault subset impact on monitored height
f_dir, g_slope = compute_worst_case_fault_direction_and_slope(
    alpha_idx,
    fault_indices,
    H,
    noise_cov,
)

# ╔═╡ b6aa2127-0f88-4dce-904a-1612c38b78cb
md"""
To get the analytic error, we choose our p-value requirement, where the default is alpha=0.05. We use this to determine the detection threshold, T. Then, we multiply the two together to get the max analytic error. 
"""

# ╔═╡ 1cc33033-eebe-4874-bf74-a46fbfc7be23
begin
	# 2. Determine the Detection Threshold (T)
	# The monitor checks if SSE < T². 
	# We need the T corresponding to our p-value requirement (alpha=0.05).
	alpha = 0.05
	dof = size(H, 1) - size(H, 2)
	T_chisq = quantile(Chisq(dof), 1 - alpha)

	# 3. Calculate Analytic Max Error (m)
	analytic_max_error = round(g_slope * T_chisq, digits=2) * m

	md"""
	### Analytic Protection Levels
	
	* **Monitored Parameter**: $(alpha_idx == 3 ? "Altitude (Height above runway)" : "State Index $alpha_idx")
	* **Analytic Max Error**: **$(analytic_max_error)**
	
	Even when the integrity check passes, there is a small chance that a "worst-case" fault is hiding just below our detection threshold. For our current configuration, we are guaranteed that any such undetected fault cannot shift our **$(alpha_idx == 3 ? "altitude" : "selected state")** by more than **$(analytic_max_error)**.
	"""
end

# ╔═╡ Cell order:
# ╟─f6bfcc72-21f3-4de4-b50f-b044b20d6e76
# ╟─e56ca9f7-9c84-4985-89a9-a6f3d60111aa
# ╟─1821382b-4378-4e78-88e9-0cd4a4a699c8
# ╠═b4fee6f9-7344-4d8c-b16e-33eff1cede68
# ╟─9612a518-b17a-467a-a6c8-6661aff7a1d8
# ╟─09c592ab-17a5-4829-a646-6edb2cb9541e
# ╠═308735e9-c796-4404-9b07-e65b3451f9be
# ╟─58a36b23-8d35-4e51-a833-dbd47914b12c
# ╠═1f00d3db-5cbb-4f72-b070-b21aab5325b8
# ╟─a29d1cf7-e04f-42cb-b83f-aae1de898cdf
# ╠═cf5b9e5a-9449-4d03-b65c-8ee34f0cdbd6
# ╠═f692b3ae-6611-4905-bd74-70779155cc9f
# ╟─78b0a026-3f8f-4af2-afea-2c451976d7f0
# ╠═afe48040-9b95-42e5-8961-9e8ecdc1662b
# ╠═6c6ccd8d-f5a7-41c4-9e16-0148a1b7462a
# ╠═c3c9e87c-4a30-4e15-a085-02d2330f795d
# ╠═60f7b871-f48d-4a95-8f7c-82022a0a4daf
# ╠═d12b5fc0-7819-444a-b297-402448b9960e
# ╠═95b02000-cb5a-4e5f-8984-61850448ec0f
# ╟─abb49dbf-be2b-442c-a6bf-f3038303f6c3
# ╠═0c865680-4ed3-4429-824b-1a93e3b30ca9
# ╟─7af47b79-8e9a-4bcf-9b30-882d5eb317c2
# ╠═c457014a-eacc-4892-a246-a4a1a2f2541b
# ╠═e04102c1-f310-46db-ae08-87bdd66ec11f
# ╟─bd2b813e-e7dc-4616-8bcd-b189a00e087b
# ╠═449bf3a9-8593-4e06-b6ea-9ecded426aa4
# ╟─0c897c1b-0e57-4292-89e1-7ee356bd1fa0
# ╠═26f751af-4fd6-4caa-8534-d2401ffa1fb4
# ╟─3fb311c4-69ce-4ef5-a9f6-88f6f7e7f473
# ╠═5839d7b4-a97f-484b-bdd6-d8446ebe9114
# ╠═25806775-15b4-4a89-852a-4e3f27bf623d
# ╠═5befb9d5-53bc-4367-9543-0c62894cea9c
# ╟─a3b64e53-8b18-44c2-84be-83802065b5b5
# ╠═1a11b8c8-7534-4f19-a072-b9eb8c94f095
# ╟─981ab774-a55b-462d-a8b0-b78e921ce620
# ╟─cd7f4a79-8d5b-4553-80ed-a80b6ed48158
# ╟─631d3b99-a287-4f4a-9494-3ddd5925a1be
# ╠═1ab2cf0b-3eae-4958-92d5-84022b733d7b
# ╟─b6aa2127-0f88-4dce-904a-1612c38b78cb
# ╠═1cc33033-eebe-4874-bf74-a46fbfc7be23
