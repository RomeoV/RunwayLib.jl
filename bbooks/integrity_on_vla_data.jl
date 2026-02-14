### A Pluto.jl notebook ###
# v0.20.21

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

# ╔═╡ 3745f4fa-fbfa-11f0-8bbf-cd24cef3b17f
import Pkg; Pkg.activate(".")

# ╔═╡ a1b2c3d4-0001-0000-0000-000000000001
begin
    using WGLMakie
    WGLMakie.activate!()
    using RunwayLib, Unitful, Unitful.DefaultSymbols, Rotations, CSV, DataFrames
    using Distributions: Normal, MvNormal, Chisq, quantile
    using LinearAlgebra: Diagonal, I
    using OhMyThreads: tmap
    # Explicit imports to resolve ambiguity with Makie and for integrity functions
    import RunwayLib: px, compute_H, compute_worst_case_fault_direction_and_slope,
		getline, comparelines, LineFeatures, NO_LINES, Line
	using PlutoUI
	using LinearAlgebra
	using StatsBase
	using RunwayLib.StaticArrays: SA
end

# ╔═╡ a1b2c3d4-0002-0000-0000-000000000002
md"""
# Pose Estimation on VLA Flight Data

This notebook processes the VLA two landings dataset (FT113, FT216) and compares RunwayLib pose estimates against ground truth. It features:
- **Trajectory selection** (FT113 vs FT216)
- **Covariance-based noise model** from per-keypoint uncertainty
- **Configurable integrity threshold** (alpha parameter)
- **Threaded trajectory processing** via OhMyThreads
"""

# ╔═╡ a1b2c3d4-0003-0000-0000-000000000003
# Helper to parse numbers with comma separators (e.g., "2,002.2" -> 2002.2)
begin
	parse_num(s::AbstractString) = parse(Float64, replace(s, "," => ""))
	parse_num(x::Number) = Float64(x)
	parse_num(::Missing) = missing
end

# ╔═╡ a1b2c3d4-0004-0000-0000-000000000004
# Load the CSV data
df_raw = CSV.read("data/VLA_two_landings.csv", DataFrame)

# ╔═╡ 510711e4-ad1b-4dc3-8a63-eca2873fd334
names(df_raw)

# ╔═╡ 7e52a693-c474-4631-af9c-c3c01767d90f
let
	row = first(eachrow(df_raw))
	row.gt_label_runway_centerline_edge_start_y_px, row.pred_edge_centerline_start_y_px,
	row.pred_edge_centerline_start_y_px
end

# ╔═╡ ae66ebfc-5949-4119-b0c7-059a1bc5434a
let
edge_names = filter(n->occursin(r"pred_edge.*_px", n), names(df_raw))
	df_raw[!, edge_names]
end

# ╔═╡ 5d0d5c3a-0429-467d-a8d9-2778febb9ff2
edge_names = filter(n->occursin(r"pred_edge.*_px", n), names(df_raw))

# ╔═╡ 68d9df40-6870-4fa0-aa91-97bf15b839b8
let
edge_names = filter(n->occursin(r"gt.*_edge_.*_px", n), names(df_raw))
	df_raw[!, edge_names]
end

# ╔═╡ a1b2c3d4-0004-0001-0000-000000000001
md"""
## Trajectory Selection

Select which flight test to analyze:
"""

# ╔═╡ a1b2c3d4-0004-0002-0000-000000000001
md"Flight Test: $(@bind selected_flight Select([\"FT113\", \"FT216\"]; default=\"FT113\"))"

# ╔═╡ a1b2c3d4-0005-0000-0000-000000000005
# Filter for selected trajectory and rows with valid keypoint detections
df = let
    pred_cols = [
        :pred_kp_bottom_left_x_px,
        :pred_kp_bottom_left_y_px,
        :pred_kp_top_left_x_px,
        :pred_kp_top_left_y_px,
        :pred_kp_top_right_x_px,
        :pred_kp_top_right_y_px,
        :pred_kp_bottom_right_x_px,
        :pred_kp_bottom_right_y_px,
    ]
    uq_cols = [
        :pred_kp_uncertainty_bottom_left_xx_px2,
        :pred_kp_uncertainty_bottom_left_yy_px2,
        :pred_kp_uncertainty_top_left_xx_px2,
        :pred_kp_uncertainty_top_left_yy_px2,
        :pred_kp_uncertainty_top_right_xx_px2,
        :pred_kp_uncertainty_top_right_yy_px2,
        :pred_kp_uncertainty_bottom_right_xx_px2,
        :pred_kp_uncertainty_bottom_right_yy_px2,
    ]

    df_flight = filter(row -> row.flight_test == selected_flight, df_raw)
	@info "`df_flight` has $(nrow(df_flight)) samples."
    df = filter(df_flight) do row
        all([pred_cols; uq_cols]) do col
            !ismissing(getindex(row, col))
        end
    end
    df = filter(df) do row
       all(uq_cols) do col
           parse_num(getindex(row, col))<5^2
       end
    end
	@info "After filtering missing predictions / uqs, and selecting only reasonable uncertainty levels (<5px of std), $(nrow(df)) samples remain."
	df
end

# ╔═╡ a1b2c3d4-0005-0001-0000-000000000001
md"Selected **$(selected_flight)**: $(nrow(df)) valid rows"

# ╔═╡ a1b2c3d4-0006-0000-0000-000000000006
md"""
## Runway Geometry Setup

The runway corners are defined in a local coordinate system where:
- X-axis points along the runway (toward far end)
- Y-axis points left (from pilot's perspective)
- Z-axis points up

Corner ordering (matching RunwayLib docs convention):
1. near_left (bottom_left) - threshold, left side
2. far_left (top_left) - far end, left side
3. far_right (top_right) - far end, right side
4. near_right (bottom_right) - threshold, right side
"""

# ╔═╡ a1b2c3d4-0007-0000-0000-000000000007
function make_runway_corners(length_m, width_m)
    half_width = width_m / 2
    SA[
        WorldPoint(0.0m, half_width * m, 0m),           # near_left (bottom_left)
        WorldPoint(length_m * m, half_width * m, 0m),   # far_left (top_left)
        WorldPoint(length_m * m, -half_width * m, 0m),  # far_right (top_right)
        WorldPoint(0.0m, -half_width * m, 0m),          # near_right (bottom_right)
    ]
end

# ╔═╡ a1b2c3d4-0008-0000-0000-000000000008
# Observed corners must match the same order as runway_corners
function extract_observed_corners(row)
    SA[
        ProjectionPoint(parse_num(row.pred_kp_bottom_left_x_px)px, parse_num(row.pred_kp_bottom_left_y_px)px),
        ProjectionPoint(parse_num(row.pred_kp_top_left_x_px)px, parse_num(row.pred_kp_top_left_y_px)px),
        ProjectionPoint(parse_num(row.pred_kp_top_right_x_px)px, parse_num(row.pred_kp_top_right_y_px)px),
        ProjectionPoint(parse_num(row.pred_kp_bottom_right_x_px)px, parse_num(row.pred_kp_bottom_right_y_px)px),
    ]
end

# ╔═╡ 4abea7de-06f4-47db-9d5a-4e7b90b898d7
parse_num.(
	df[!, [:pred_kp_uncertainty_bottom_left_xx_px2, 
		   :pred_kp_uncertainty_top_left_xx_px2    ]]
)

# ╔═╡ aa29f607-c8fb-4ba7-8957-c75de18546de
let
    fig = Figure()
	Label(fig[0, 1:2], text=L"Pixel Uncertainties (1 $\sigma$)")
    ax_nl = Axis(fig[2, 1], title="Near Left")
    hist!(ax_nl, sqrt.(parse_num.(df[!, :pred_kp_uncertainty_bottom_left_xx_px2])), label="x", alpha=0.6)
    hist!(ax_nl, sqrt.(parse_num.(df[!, :pred_kp_uncertainty_bottom_left_yy_px2])), label="y", alpha=0.6)

    ax_fl = Axis(fig[1, 1], title="Far Left")
    hist!(ax_fl, sqrt.(parse_num.(df[!, :pred_kp_uncertainty_top_left_xx_px2])), label="x", alpha=0.6)
    hist!(ax_fl, sqrt.(parse_num.(df[!, :pred_kp_uncertainty_top_left_yy_px2])), label="y", alpha=0.6)


    ax_fr = Axis(fig[1, 2], title="Far Right")
    hist!(ax_fr, sqrt.(parse_num.(df[!, :pred_kp_uncertainty_top_right_xx_px2])), label="x", alpha=0.6)
    hist!(ax_fr, sqrt.(parse_num.(df[!, :pred_kp_uncertainty_top_right_yy_px2])), label="y", alpha=0.6)


    ax_nr = Axis(fig[2, 2], title="Near Right")
    hist!(ax_nr, sqrt.(parse_num.(df[!, :pred_kp_uncertainty_bottom_right_xx_px2])), label="x", alpha=0.6)
    hist!(ax_nr, sqrt.(parse_num.(df[!, :pred_kp_uncertainty_bottom_right_yy_px2])), label="y", alpha=0.6)

	axislegend.([ax_nl, ax_fl, ax_fr, ax_nr])

    fig
end


# ╔═╡ a1b2c3d4-0008-0001-0000-000000000001
md"""
## Uncertainty Model

This dataset provides per-keypoint 2x2 covariance matrices. We build a full 8x8
block-diagonal covariance from these individual covariances.
"""

# ╔═╡ a1b2c3d4-0008-0002-0000-000000000001
# Build noise model from covariance data in CSV
function make_noise_model_from_covariance(row)
    # Corner order: bottom_left, top_left, top_right, bottom_right
    corners = [:bottom_left, :top_left, :top_right, :bottom_right]

    # Build 8x8 block-diagonal covariance matrix
    cov_matrix = zeros(8, 8)
    for (i, corner) in enumerate(corners)
        xx = parse_num(row[Symbol("pred_kp_uncertainty_$(corner)_xx_px2")])
        xy = parse_num(row[Symbol("pred_kp_uncertainty_$(corner)_xy_px2")])
        yy = parse_num(row[Symbol("pred_kp_uncertainty_$(corner)_yy_px2")])

        # Each corner occupies 2 rows/cols (x, y)
        idx = 2*(i-1) + 1
        cov_matrix[idx, idx] = xx
        # for now don't consider correlations.
		#cov_matrix[idx, idx+1] = xy
        #cov_matrix[idx+1, idx] = xy
        cov_matrix[idx+1, idx+1] = yy
    end

    CorrGaussianNoiseModel(MvNormal(zeros(8), cov_matrix))
end

# ╔═╡ a1b2c3d4-0008-0003-0000-000000000003
# Fallback: Build noise model from confidence scores (same order as corners)
function make_noise_model_from_confidence(row, global_sigma)
    confidences = [
        row.pred_kp_bottom_left_conf,
        row.pred_kp_top_left_conf,
        row.pred_kp_top_right_conf,
        row.pred_kp_bottom_right_conf,
    ]
    # sigma = global_sigma * confidence for each corner
    # Each corner has 2 measurements (x, y), so repeat sigma twice per corner
    distributions = [Normal(0.0, global_sigma * conf) for conf in confidences for _ in 1:2]
    UncorrGaussianNoiseModel(distributions)
end

# ╔═╡ a1b2c3d4-0024-0000-0000-000000000001
# Compute protection levels for integrity monitoring
# Uses single-fault hypothesis (one measurement at a time) to avoid numerical issues
function compute_protection_levels(cam_pos, cam_rot, pf::PointFeatures;
								   alpha=0.05, dof=3)
    H = compute_H(cam_pos, cam_rot, pf)[:, 1:dof]
    noise_cov = Matrix(pf.cov)
    n_meas = size(H, 1)

    # Chi-squared threshold for given confidence level
    dofs = n_meas - dof
    dofs <= 0 && error("Not enough measurements")
    chi2_thresh = quantile(Chisq(dofs), 1 - alpha)

    # Compute max protection level over all single-fault hypotheses
    bounds = map(1:3) do alpha_idx  # alongtrack, crosstrack, altitude
		fault_hypotheses = (1:2, 3:4, 5:6, 7:8)
        max_pl = maximum(fault_hypotheses) do fault_idx
			_, g_slope = compute_worst_case_fault_direction_and_slope(
				alpha_idx, fault_idx, Matrix(H), noise_cov
			)
			abs(g_slope) * sqrt(chi2_thresh)
        end
    end
    (alongtrack=bounds[1], crosstrack=bounds[2], altitude=bounds[3])
end

# ╔═╡ a1b2c3d4-0009-0000-0000-000000000009
md"""
## Single Pose Estimate Example

Let's find a row with good quality data (low pipeline error) and compare our estimate.
"""

# ╔═╡ a1b2c3d4-0010-0000-0000-000000000010
# ╠═╡ disabled = true
#=╠═╡
# Filter for rows with valid and reasonable along_track_error
df_good = let
    df_with_errors = filter(row -> !ismissing(row.along_track_error_m) &&
                                   row.along_track_error_m != "1,000,000", df)
    errors = [parse_num(row.along_track_error_m) for row in eachrow(df_with_errors)]
    good_mask = abs.(errors) .< 100  # rows where pipeline error < 100m
    df_with_errors[good_mask, :]
end
  ╠═╡ =#

# ╔═╡ b2958518-1c72-4ed4-9a8b-cf2f271db480
df_good = df

# ╔═╡ a1b2c3d4-0011-0000-0000-000000000011
# Select a sample row
row = df_good[1, :];

# ╔═╡ a1b2c3d4-0012-0000-0000-000000000012
# Parse runway geometry and create corners
begin
    runway_length = parse_num(row.active_runway_length_m)
    runway_width = parse_num(row.active_runway_width_m)
    runway_corners = make_runway_corners(runway_length, runway_width)
end

# ╔═╡ a1b2c3d4-0013-0000-0000-000000000013
# Extract observed keypoints
observed_corners = extract_observed_corners(row)

# ╔═╡ a1b2c3d4-0016-0000-0000-000000000016
md"""
## Comparison to Ground Truth

The coordinate mapping:
- `cam_pos_est.x` ↔ `gt_along_track_distance_m` (distance along runway centerline)
- `cam_pos_est.y` ↔ `gt_cross_track_distance_m` (lateral offset)
- `cam_pos_est.z` ↔ `gt_height_m` (altitude above runway)
"""

# ╔═╡ a1b2c3d4-0017-0000-0000-000000000017
# Ground truth from CSV
gt = (
    along_track = parse_num(row.gt_along_track_distance_m),
    cross_track = row.gt_cross_track_distance_m,
    height = row.gt_height_m,
    pitch = row.gt_pitch_deg,
    roll = row.gt_roll_deg,
    yaw = row.gt_yaw_deg,
)

# ╔═╡ a1b2c3d4-0018-0000-0000-000000000018
# Pipeline's prediction for reference
pipeline = (
    along_track = parse_num(row.pred_along_track_distance_m),
    cross_track = parse_num(row.pred_cross_track_distance_m),
    height = parse_num(row.pred_height_m),
    error = parse_num(row.along_track_error_m),
)

# ╔═╡ a1b2c3d4-0021-0000-0000-000000000021
md"""
### Rotation Comparison

Note: Rotation conventions between the CSV ground truth and RunwayLib may differ.
The CSV uses a specific aircraft/navigation convention while RunwayLib uses RotZYX.
Further investigation needed to establish exact mapping.
"""

# ╔═╡ dca746d9-5a4a-4386-830f-0bada1377367
md"# Unit-free protection level"

# ╔═╡ 2e86ab6e-0949-11f1-a7b4-430f374d1f1a
md"""
## Plan: using edge features for integrity

### What we have
The dataset provides **edge features** for 5 runway edges (bottom, top, left, right, centerline).
Each edge has 3 points: `point1`, `mid`, `point2`. The GT equivalents are `start`, `mid`, `stop`.

### What we learned
These edge features define **lines**, not points with known 3D correspondences:
- `mid` = where the edge line crosses the **image center** (not the projection of the 3D midpoint)
  - bottom/top: x is always 2047.5 (image center x); y varies
  - left/right: y is always 1499.5 (image center y); x varies
- `point1`/`point2` = where the edge line hits the **image boundary** (x=0, x=4095, y=0, y=2999)

⚠️ This means we **cannot** assign a fixed `WorldPoint` to any edge feature, because the 3D
point that projects to e.g. the "bottom mid" pixel depends on the camera pose (which is what
we're estimating).

### Error statistics (pred vs GT, informative coordinate only)
| Edge mid | Std | RMS | 95th pctl |
|----------|-----|-----|-----------|
| bottom (y) | 10.6 px | 13.4 px | 26.9 px |
| top (y) | 9.4 px | 11.5 px | 24.0 px |
| left (x) | 2.9 px | 2.9 px | 5.6 px |
| right (x) | 21.3 px | 21.3 px | 11.6 px |

Pooled sigma ≈ 13 px. There's also a positive bias of ~7 px in bottom/top y.

### Options to use edge features for integrity
1. **Line features** — extend `compute_H` to handle the constraint that a known 3D line must
   project onto the detected image line. Each edge gives 1 constraint (the line's angle/offset).
   This is the theoretically correct approach but requires extending RunwayLib.
2. **Derived corner points** — intersect pairs of edge lines in the image to recover the 4
   corner positions via an independent method. These intersections DO have known 3D coordinates
   (the corner keypoints). This gives redundant corner measurements from a different detector,
   which is useful for integrity (independent fault modes).
3. **Hybrid** — use the edge lines to refine/validate the corner keypoints, then feed the
   validated corners into the existing `PointFeatures` framework.

### Resolution ✓
**Chose Option 1**: RunwayLib already has native `LineFeatures` support (Hough parametrization).
Edge lines are now modeled as `LineFeatures` with `getline(point1, point2)` from CSV predictions
and world line endpoints defined by runway geometry. Each line adds 2 effective DOF (despite 3
residual components, since cos²θ + sin²θ = 1).
"""

# ╔═╡ b6f1746a-0945-11f1-8bc5-d36f304c5851
# Plot predicted vs GT edge features for the example row
let
	col_names = names(df_raw)
	edges = ["bottom", "top", "left", "right", "centerline"]
	edge_colors = Dict(
		"bottom" => :red, "top" => :blue, "left" => :green,
		"right" => :orange, "centerline" => :purple
	)

	fig = Figure(size=(1200, 900))
	ax = Axis(fig[1, 1],
		title="Predicted vs GT Edge Features (example row)",
		xlabel="x [px]", ylabel="y [px]",
		yreversed=true
	)

	# Plot predicted edges (solid lines, filled markers)
	for edge in edges
		xs, ys = Float64[], Float64[]
		for pt in ["point1", "mid", "point2"]
			x_col = "pred_edge_$(edge)_$(pt)_x_px"
			y_col = "pred_edge_$(edge)_$(pt)_y_px"
			x_col in col_names || continue
			x, y = parse_num(row[Symbol(x_col)]), parse_num(row[Symbol(y_col)])
			(isnan(x) || isnan(y)) && continue
			push!(xs, x); push!(ys, y)
		end
		isempty(xs) && continue
		lines!(ax, xs, ys, color=(edge_colors[edge], 0.5), linewidth=2)
		scatter!(ax, xs, ys, color=edge_colors[edge], markersize=12,
			label="pred $edge")
	end

	# Plot GT edges (dashed lines, open markers)
	for edge in edges
		xs, ys = Float64[], Float64[]
		for pt in ["start", "mid", "stop"]
			x_col = "gt_label_runway_$(edge)_edge_$(pt)_x_px"
			y_col = "gt_label_runway_$(edge)_edge_$(pt)_y_px"
			x_col in col_names || continue
			x, y = parse_num(row[Symbol(x_col)]), parse_num(row[Symbol(y_col)])
			(isnan(x) || isnan(y)) && continue
			push!(xs, x); push!(ys, y)
		end
		isempty(xs) && continue
		lines!(ax, xs, ys, color=(edge_colors[edge], 0.3), linewidth=3, linestyle=:dash)
		scatter!(ax, xs, ys, color=:white, strokecolor=edge_colors[edge],
			strokewidth=2, markersize=14, label="gt $edge")
	end

	# Plot the 4 corner keypoints
	kp_names = ["bottom_left", "top_left", "top_right", "bottom_right"]
	kp_xs = [parse_num(row[Symbol("pred_kp_$(k)_x_px")]) for k in kp_names]
	kp_ys = [parse_num(row[Symbol("pred_kp_$(k)_y_px")]) for k in kp_names]
	scatter!(ax, kp_xs, kp_ys, color=:black, marker=:star5, markersize=20, label="pred corners")
	lines!(ax, [kp_xs; kp_xs[1]], [kp_ys; kp_ys[1]],
		color=(:black, 0.3), linewidth=1, linestyle=:dash)

	Legend(fig[1, 2], ax, framevisible=false, unique=true)
	fig
end

# ╔═╡ 0c38d900-0947-11f1-bbb0-000000000001
# Pred vs GT line errors in Hough (r, θ) space for the example row
let
	col_names = names(df_raw)
	edges = ["bottom", "top", "left", "right", "centerline"]
	rows_md = String[]

	for edge in edges
		pred_cols = ["pred_edge_$(edge)_point1_x_px", "pred_edge_$(edge)_point1_y_px",
					 "pred_edge_$(edge)_point2_x_px", "pred_edge_$(edge)_point2_y_px"]
		gt_cols = ["gt_label_runway_$(edge)_edge_start_x_px", "gt_label_runway_$(edge)_edge_start_y_px",
				   "gt_label_runway_$(edge)_edge_stop_x_px", "gt_label_runway_$(edge)_edge_stop_y_px"]
		all(c -> c in col_names, vcat(pred_cols, gt_cols)) || continue

		vals = [parse_num(row[Symbol(c)]) for c in vcat(pred_cols, gt_cols)]
		if any(isnan, vals)
			push!(rows_md, "| $edge | — | — | — | — | — | — |")
			continue
		end

		pred_l = getline(ProjectionPoint(vals[1]*px, vals[2]*px), ProjectionPoint(vals[3]*px, vals[4]*px))
		gt_l = getline(ProjectionPoint(vals[5]*px, vals[6]*px), ProjectionPoint(vals[7]*px, vals[8]*px))
		resid = comparelines(pred_l, gt_l)

		pr = round(ustrip(px, pred_l.r), digits=1)
		pt = round(rad2deg(ustrip(rad, pred_l.theta)), digits=2)
		gr = round(ustrip(px, gt_l.r), digits=1)
		ga = round(rad2deg(ustrip(rad, gt_l.theta)), digits=2)
		dr = round(resid[1], digits=1)
		# Δθ in degrees from the cos/sin residual (approximate)
		dθ = round(rad2deg(asin(clamp(resid[3], -1, 1))), digits=3)

		push!(rows_md, "| $edge | $pr | $gr | **$dr** | $(pt)° | $(ga)° | **$(dθ)°** |")
	end

	tbl = join([
		"### Pred vs GT edge lines — example row (Hough representation)",
		"",
		"| Edge | pred r [px] | GT r [px] | Δr [px] | pred θ | GT θ | Δθ |",
		"|------|------------|----------|---------|--------|------|-----|",
		rows_md...,
		"",
		"*Δr from `comparelines` residual (with π-ambiguity resolution). Δθ approximated from sin-component.*",
	], "\n")
	Markdown.parse(tbl)
end

# ╔═╡ f640d1b4-ac63-4110-96d5-559287b43ae7
function unitfree_protection_level(cam_pos, cam_rot, pf, lf=NO_LINES;
		alphaidx=1, fault_indices=1:2, alpha=0.05)
	# Combined covariance (block diagonal of point + line covariances)
	n_lines = length(lf.world_line_endpoints)
	Σy = cat(pf.cov, lf.cov, dims=(1,2)) |> Matrix
	@assert issymmetric(Σy)

	# Combined Jacobian (RunwayLib dispatches correctly for NO_LINES)
	H = compute_H(cam_pos, cam_rot, pf, lf)

	Ly = cholesky(Σy).L
	Σx = inv(H' * (Σy \ H)) |> hermitianpart
	@assert issymmetric(Σx)

	n_meas = size(H, 1)
	Hbow = inv(Ly) * H * Diagonal(sqrt.(diag(Σx)))
	_, g = compute_worst_case_fault_direction_and_slope(
		alphaidx, fault_indices, Hbow, I(n_meas)
	)

	# DOF: 2 per point + 2 per line. Lines have 3 residual components [Δr, Δcosθ, Δsinθ]
	# but only 2 true DOF because cos²θ + sin²θ = 1 constrains the representation.
	# Adding a correlated measurement doesn't add a degree of freedom.
	n_pts = length(pf.runway_corners)
	dof = (2n_pts + 2n_lines) - size(H, 2)
	num_stds_y = sqrt(quantile(Chisq(dof), 1-alpha))
	std_x = sqrt(Σx[alphaidx, alphaidx])
	return g * std_x * num_stds_y
end


# ╔═╡ 0c38cd0a-0947-11f1-8be0-afa504b38a75
md"""
### Feature selection for unitfree integrity

The 4 corner keypoints are always included as **point features** (2 DOF each).
Edge lines are optional **line features** (2 DOF each, represented in Hough form).

Each edge line is defined by two `WorldPoint` endpoints and observed via `getline(point1, point2)` from the CSV predictions.

| Edge line | World endpoints | DOF |
|-----------|----------------|-----|
| bottom | `(0,+W/2,0)` → `(0,-W/2,0)` | 2 |
| top | `(L,+W/2,0)` → `(L,-W/2,0)` | 2 |
| left | `(0,+W/2,0)` → `(L,+W/2,0)` | 2 |
| right | `(0,-W/2,0)` → `(L,-W/2,0)` | 2 |
| centerline | `(0,0,0)` → `(L,0,0)` | 2 |

Corners (always on): $(@bind use_corners CheckBox(default=true))

Edge lines: $(@bind use_bottom_line CheckBox()) bottom  $(@bind use_top_line CheckBox()) top  $(@bind use_left_line CheckBox()) left  $(@bind use_right_line CheckBox()) right  $(@bind use_cl_line CheckBox()) centerline

Line noise σ\_r [px] (σ\_cosθ, σ\_sinθ derived from line angle): $(@bind line_noise_scale Slider(5.0:5.0:200.0; show_value=true, default=30.0))
"""

# ╔═╡ 0c38cf96-0947-11f1-90b5-43dc33093ab5
# Build PointFeatures (corners) + LineFeatures (edges) from checkbox selection
unitfree_pf, unitfree_lf, feature_status = let
	half_w = runway_width / 2
	L = runway_length
	status = String[]

	# --- Point features: 4 corners (always included) ---
	wps = [
		WorldPoint(0.0m, half_w*m, 0m),       # near_left (bottom_left)
		WorldPoint(L*m, half_w*m, 0m),         # far_left (top_left)
		WorldPoint(L*m, -half_w*m, 0m),        # far_right (top_right)
		WorldPoint(0.0m, -half_w*m, 0m),       # near_right (bottom_right)
	]
	ops = [
		ProjectionPoint(parse_num(row.pred_kp_bottom_left_x_px)px, parse_num(row.pred_kp_bottom_left_y_px)px),
		ProjectionPoint(parse_num(row.pred_kp_top_left_x_px)px, parse_num(row.pred_kp_top_left_y_px)px),
		ProjectionPoint(parse_num(row.pred_kp_top_right_x_px)px, parse_num(row.pred_kp_top_right_y_px)px),
		ProjectionPoint(parse_num(row.pred_kp_bottom_right_x_px)px, parse_num(row.pred_kp_bottom_right_y_px)px),
	]
	n_pts = 4
	n_meas_pts = 2n_pts
	cov_diag = zeros(n_meas_pts)
	for i in 1:4
		xx = parse_num(row[Symbol("pred_kp_uncertainty_$([:bottom_left, :top_left, :top_right, :bottom_right][i])_xx_px2")])
		yy = parse_num(row[Symbol("pred_kp_uncertainty_$([:bottom_left, :top_left, :top_right, :bottom_right][i])_yy_px2")])
		cov_diag[2i-1] = xx
		cov_diag[2i] = yy
	end
	noise_pts = CorrGaussianNoiseModel(MvNormal(zeros(n_meas_pts), Diagonal(cov_diag)))
	pf = PointFeatures(wps, ops, CAMERA_CONFIG_OFFSET, noise_pts)
	push!(status, "4 corners: included (point features, 8 meas)")

	# --- Line features: edges ---
	# Each edge defined by (use_flag, name, world_endpoint_1, world_endpoint_2, pt1_x_col, pt1_y_col, pt2_x_col, pt2_y_col)
	edge_defs = [
		(use_bottom_line, "bottom",
			WorldPoint(0.0m, half_w*m, 0m), WorldPoint(0.0m, -half_w*m, 0m),
			"pred_edge_bottom_point1_x_px", "pred_edge_bottom_point1_y_px",
			"pred_edge_bottom_point2_x_px", "pred_edge_bottom_point2_y_px"),
		(use_top_line, "top",
			WorldPoint(L*m, half_w*m, 0m), WorldPoint(L*m, -half_w*m, 0m),
			"pred_edge_top_point1_x_px", "pred_edge_top_point1_y_px",
			"pred_edge_top_point2_x_px", "pred_edge_top_point2_y_px"),
		(use_left_line, "left",
			WorldPoint(0.0m, half_w*m, 0m), WorldPoint(L*m, half_w*m, 0m),
			"pred_edge_left_point1_x_px", "pred_edge_left_point1_y_px",
			"pred_edge_left_point2_x_px", "pred_edge_left_point2_y_px"),
		(use_right_line, "right",
			WorldPoint(0.0m, -half_w*m, 0m), WorldPoint(L*m, -half_w*m, 0m),
			"pred_edge_right_point1_x_px", "pred_edge_right_point1_y_px",
			"pred_edge_right_point2_x_px", "pred_edge_right_point2_y_px"),
		(use_cl_line, "centerline",
			WorldPoint(0.0m, 0.0m, 0m), WorldPoint(L*m, 0.0m, 0m),
			"pred_edge_centerline_point1_x_px", "pred_edge_centerline_point1_y_px",
			"pred_edge_centerline_point2_x_px", "pred_edge_centerline_point2_y_px"),
	]

	# Collect selected edges (filter by checkbox and NaN)
	selected_edges = filter(edge_defs) do (use, name, wp1, wp2, x1c, y1c, x2c, y2c)
		use || return false
		vals = [parse_num(row[Symbol(c)]) for c in (x1c, y1c, x2c, y2c)]
		if any(isnan, vals)
			push!(status, "$name line: SKIPPED (NaN in data)")
			return false
		end
		push!(status, "$name line: included")
		return true
	end

	if !isempty(selected_edges)
		world_endpoints = [(wp1, wp2) for (_, _, wp1, wp2, _...) in selected_edges]
		observed_lines = [
			let vals = [parse_num(row[Symbol(c)]) for c in (x1c, y1c, x2c, y2c)]
				getline(ProjectionPoint(vals[1]*px, vals[2]*px),
						ProjectionPoint(vals[3]*px, vals[4]*px))
			end
			for (_, _, _, _, x1c, y1c, x2c, y2c) in selected_edges
		]
		# Per-line covariance: 3 diagonal entries [σ²_r, σ²_cosθ, σ²_sinθ]
		# The angular sigmas depend on the line angle θ because:
		#   Δcosθ ≈ -sin(θ₀)·Δθ  and  Δsinθ ≈ cos(θ₀)·Δθ
		# So σ_cosθ = |sin(θ₀)| · σ_θ  and  σ_sinθ = |cos(θ₀)| · σ_θ
		# where σ_θ ≈ σ_r / line_length_px (endpoint noise → angle noise).
		σ_r = line_noise_scale  # px
		ε_floor = 1e-6  # floor to avoid singular covariance
		line_cov_diag = Float64[]
		for line in observed_lines
			θ = ustrip(rad, line.theta)
			# Estimate line length from r and image size (~4000px diagonal)
			line_len = 3000.0  # approximate effective line length in px
			σ_θ = σ_r / line_len
			σ_cosθ = max(abs(sin(θ)) * σ_θ, ε_floor)
			σ_sinθ = max(abs(cos(θ)) * σ_θ, ε_floor)
			append!(line_cov_diag, [σ_r^2, σ_cosθ^2, σ_sinθ^2])
		end
		line_cov = Diagonal(line_cov_diag)
		lf = LineFeatures(world_endpoints, observed_lines, CAMERA_CONFIG_OFFSET, line_cov)
	else
		lf = NO_LINES
	end

	(pf, lf, status)
end

# ╔═╡ 0c38d800-0947-11f1-aaa0-000000000001
# Compute pred vs GT line residuals across all rows to estimate noise sigma
edge_line_errors = let
	edges = ["bottom", "top", "left", "right", "centerline"]
	col_names = names(df_raw)
	results = Dict{String, Vector{Vector{Float64}}}()

	for edge in edges
		# Pred columns use point1/point2; GT columns use start/stop
		pred_cols = [
			"pred_edge_$(edge)_point1_x_px", "pred_edge_$(edge)_point1_y_px",
			"pred_edge_$(edge)_point2_x_px", "pred_edge_$(edge)_point2_y_px",
		]
		gt_cols = [
			"gt_label_runway_$(edge)_edge_start_x_px", "gt_label_runway_$(edge)_edge_start_y_px",
			"gt_label_runway_$(edge)_edge_stop_x_px", "gt_label_runway_$(edge)_edge_stop_y_px",
		]
		all(c -> c in col_names, vcat(pred_cols, gt_cols)) || continue

		resids = Vector{Float64}[]
		for r in eachrow(df_raw)
			raw = [parse_num(r[Symbol(c)]) for c in vcat(pred_cols, gt_cols)]
			any(ismissing, raw) && continue
			vals = Float64.(raw)
			any(!isfinite, vals) && continue
			pred_line = getline(
				ProjectionPoint(vals[1]*px, vals[2]*px),
				ProjectionPoint(vals[3]*px, vals[4]*px))
			gt_line = getline(
				ProjectionPoint(vals[5]*px, vals[6]*px),
				ProjectionPoint(vals[7]*px, vals[8]*px))
			push!(resids, collect(comparelines(pred_line, gt_line)))
		end
		results[edge] = resids
	end

	# Build summary table
	header = "| Edge | N | σ(Δr) | σ(Δcos θ) | σ(Δsin θ) | RMS |"
	sep = "|------|---|-------|-----------|-----------|-----|"
	rows = String[]
	for edge in edges
		haskey(results, edge) || continue
		resids = results[edge]
		isempty(resids) && continue
		mat = reduce(hcat, resids)'  # N×3
		stds = [std(mat[:, j]) for j in 1:3]
		rms = sqrt(mean(sum(mat.^2, dims=2)))
		push!(rows, "| $edge | $(size(mat,1)) | $(round(stds[1], digits=3)) | $(round(stds[2], digits=4)) | $(round(stds[3], digits=4)) | $(round(rms, digits=3)) |")
	end

	Markdown.parse(join([
		"### Edge line error statistics (pred vs GT, Hough residuals)",
		"",
		header,
		sep,
		rows...,
		"",
		"**Note**: Δr is in px, Δcos θ and Δsin θ are unitless. The `comparelines` residual is `[r₁-r₂, cos θ₁ - cos θ₂, sin θ₁ - sin θ₂]` with π-ambiguity resolution.",
	], "\n"))
end

# ╔═╡ 8789a9ec-0919-4b67-9261-98ba05100f87
let pf = PointFeatures(runway_corners, observed_corners, 
				  CAMERA_CONFIG_OFFSET, CorrGaussianNoiseModel(MvNormal(1.0*I(8))))
	cond(pf.cov)
end

# ╔═╡ a1b2c3d4-0023-0000-0000-000000000001
md"""
## Trajectory Error Analysis

Process all rows in a trajectory using threaded computation.
"""

# ╔═╡ a1b2c3d4-0023-0000-0000-000000000002
md"""
### Configuration

Noise model: $(@bind noise_source Select(["covariance" => "Covariance from CSV", "confidence" => "Confidence * sigma"]; default="covariance"))

$(noise_source == "confidence" ? md"global sigma: $(@bind global_sigma Slider(1:10; show_value=true, default=5))" : md"")

alpha (integrity threshold): $(@bind alpha_val Select([0.01 => "0.01 (99%)", 0.05 => "0.05 (95%)", 0.1 => "0.1 (90%)", 0.2 => "0.2 (80%)"]; default=0.05))
"""

# ╔═╡ a1b2c3d4-0014-0000-0000-000000000014
# Build noise model based on selection
noise_model = if noise_source == "covariance"
    make_noise_model_from_covariance(row)
else
    make_noise_model_from_confidence(row, global_sigma)
end

# ╔═╡ a1b2c3d4-0014-0001-0000-000000000001
# Run pose estimation using PointFeatures with noise model
result = estimatepose6dof(PointFeatures(runway_corners, observed_corners, CAMERA_CONFIG_OFFSET, noise_model))

# ╔═╡ a1b2c3d4-0015-0000-0000-000000000015
# Extract estimated pose
cam_pos_est, cam_rot_est = result[:pos], result[:rot]

# ╔═╡ a081bc6a-1b1c-4bc3-83a2-36cfc1db5b2d
H = compute_H(cam_pos_est, cam_rot_est, runway_corners)

# ╔═╡ a43924ff-7eab-4afd-a940-46333a38c35f
pinv(H)

# ╔═╡ a1b2c3d4-0019-0000-0000-000000000019
# Our estimate
est = (
    along_track = ustrip(m, cam_pos_est.x),
    cross_track = ustrip(m, cam_pos_est.y),
    height = ustrip(m, cam_pos_est.z),
)

# ╔═╡ a1b2c3d4-0020-0000-0000-000000000020
# Position comparison
let
    println("=== Position Comparison ===")
    println("                  Along-track    Cross-track    Height")
    println("Ground Truth:     $(round(gt.along_track, digits=1))m    $(round(gt.cross_track, digits=1))m    $(round(gt.height, digits=1))m")
    println("Pipeline Pred:    $(round(pipeline.along_track, digits=1))m    $(round(pipeline.cross_track, digits=1))m    $(round(pipeline.height, digits=1))m")
    println("Our Estimate:     $(round(est.along_track, digits=1))m    $(round(est.cross_track, digits=1))m    $(round(est.height, digits=1))m")
    println()
    println("=== Errors ===")
    println("Pipeline error:   $(round(pipeline.error, digits=2))m")
    println("Our error:        $(round(est.along_track - gt.along_track, digits=2))m (along-track)")
end

# ╔═╡ a1b2c3d4-0022-0000-0000-000000000022
# Rotation comparison (convention mapping TBD)
let
    import Rotations: params
    (yaw, pitch, roll) = params(cam_rot_est) .|> rad2deg

    println("=== Rotation Comparison ===")
    println("                  Pitch       Roll        Yaw")
    println("Ground Truth:     $(round(gt.pitch, digits=2))°    $(round(gt.roll, digits=2))°    $(round(gt.yaw, digits=2))°")
    println("Our Estimate:     $(round(pitch, digits=2))°    $(round(roll, digits=2))°    $(round(yaw, digits=2))°")
    println()
    println("Note: Rotation convention mapping needs verification")
end

# ╔═╡ 0c38d460-0947-11f1-8a70-0f606135e992
# Compute unitfree protection levels for all 3 axes
unitfree_pls = let
	pf = unitfree_pf
	lf = unitfree_lf
	n_pts = length(pf.runway_corners)
	n_lines = length(lf.world_line_endpoints)
	n_meas_rows = 2n_pts + 3n_lines  # rows of H
	dof = (2n_pts + 2n_lines) - 6     # effective DOF (lines have 2 true DOF each)

	# Fault hypotheses: 2 measurements per point, 3 per line
	pt_faults = [2i-1:2i for i in 1:n_pts]
	line_offset = 2n_pts
	line_faults = [line_offset + 3j-2 : line_offset + 3j for j in 1:n_lines]
	fault_hypotheses = vcat(pt_faults, line_faults)

	pls = map(1:3) do alphaidx
		maximum(fault_hypotheses) do fidx
			unitfree_protection_level(cam_pos_est, cam_rot_est, pf, lf;
				alphaidx, fault_indices=fidx, alpha=0.05)
		end
	end

	md"""
**Unitfree protection levels** ($(n_pts) points + $(n_lines) lines, $(n_meas_rows) residuals, **dof = $(dof)**):

| Axis | Protection Level |
|------|-----------------|
| Along-track (x) | $(round(pls[1], digits=2)) m |
| Cross-track (y) | $(round(pls[2], digits=2)) m |
| Altitude (z) | $(round(pls[3], digits=2)) m |

**Features used:**
$(join(["- " * s for s in feature_status], "\n"))
"""
end

# ╔═╡ 0c38d674-0947-11f1-990c-e7feb7e1db9f
# Bar plot of protection levels for the 3 axes
let
	pf = unitfree_pf
	lf = unitfree_lf
	n_pts = length(pf.runway_corners)
	n_lines = length(lf.world_line_endpoints)
	dof = (2n_pts + 2n_lines) - 6

	pt_faults = [2i-1:2i for i in 1:n_pts]
	line_offset = 2n_pts
	line_faults = [line_offset + 3j-2 : line_offset + 3j for j in 1:n_lines]
	fault_hypotheses = vcat(pt_faults, line_faults)

	pls = map(1:3) do alphaidx
		maximum(fault_hypotheses) do fidx
			unitfree_protection_level(cam_pos_est, cam_rot_est, pf, lf;
				alphaidx, fault_indices=fidx, alpha=0.05)
		end
	end

	fig = Figure(size=(600, 400))
	ax = Axis(fig[1, 1],
		title="Unitfree Protection Levels ($(n_pts) pts + $(n_lines) lines, dof=$(dof))",
		xlabel="Axis", ylabel="Protection Level [m]",
		xticks=(1:3, ["Along-track", "Cross-track", "Altitude"])
	)
	barplot!(ax, 1:3, pls, color=[:steelblue, :orange, :green])
	fig
end

# ╔═╡ b66413ea-8f64-4016-a762-33b6795ecefa
unitfree_protection_level(
	cam_pos_est, cam_rot_est, 
	PointFeatures(runway_corners, observed_corners, 
				  CAMERA_CONFIG_OFFSET, CorrGaussianNoiseModel(MvNormal(0.5*I(8))));
	alphaidx=3, alpha=0.05
)

# ╔═╡ 21a2f0ae-77ec-4190-828a-ba8aa4e3d7a5
cov(noise_model.noisedistribution) |> diag

# ╔═╡ a1b2c3d4-0023-0000-0000-000000000003
# Process trajectory - estimate pose for each valid row, including integrity info
# Uses threading via OhMyThreads for parallel processing
function process_trajectory_threaded(df_traj; use_covariance=true, global_sigma=5.0, alpha=0.05)
    rows = collect(eachrow(df_traj))

    results = tmap(rows) do row
        try
            rwy_l = parse_num(row.active_runway_length_m)
            rwy_w = parse_num(row.active_runway_width_m)
            half_w = rwy_w / 2

            runway_corners = [
                WorldPoint(0.0m, half_w * m, 0m),
                WorldPoint(rwy_l * m, half_w * m, 0m),
                WorldPoint(rwy_l * m, -half_w * m, 0m),
                WorldPoint(0.0m, -half_w * m, 0m),
            ]

            observed_corners = [
                ProjectionPoint(parse_num(row.pred_kp_bottom_left_x_px)px, parse_num(row.pred_kp_bottom_left_y_px)px),
                ProjectionPoint(parse_num(row.pred_kp_top_left_x_px)px, parse_num(row.pred_kp_top_left_y_px)px),
                ProjectionPoint(parse_num(row.pred_kp_top_right_x_px)px, parse_num(row.pred_kp_top_right_y_px)px),
                ProjectionPoint(parse_num(row.pred_kp_bottom_right_x_px)px, parse_num(row.pred_kp_bottom_right_y_px)px),
            ]

            noise_model = if use_covariance
                make_noise_model_from_covariance(row)
            else
                make_noise_model_from_confidence(row, global_sigma)
            end

            pf = PointFeatures(runway_corners, observed_corners, CAMERA_CONFIG_OFFSET, noise_model)
            result = estimatepose6dof(pf)

            cam_pos = result[:pos]
            cam_rot = result[:rot]

            gt_along = parse_num(row.gt_along_track_distance_m)
            gt_cross = row.gt_cross_track_distance_m
            gt_height = row.gt_height_m

            est_along = ustrip(m, cam_pos.x)
            est_cross = ustrip(m, cam_pos.y)
            est_height = ustrip(m, cam_pos.z)

            # Compute integrity statistic
            integrity = compute_integrity_statistic(cam_pos, cam_rot, pf)
            passes = integrity.p_value > alpha

            # Compute protection levels
            pl = compute_protection_levels(cam_pos, cam_rot, pf; alpha=alpha, dof=6)

            # Check if GT is within bounds
            gt_within_along = abs(est_along - gt_along) <= pl.alongtrack
            gt_within_cross = abs(est_cross - gt_cross) <= pl.crosstrack
            gt_within_alt = abs(est_height - gt_height) <= pl.altitude

            # Extract corner covariances (trace = xx + yy as uncertainty magnitude)
            cov_bl = parse_num(row.pred_kp_uncertainty_bottom_left_xx_px2) + parse_num(row.pred_kp_uncertainty_bottom_left_yy_px2)
            cov_tl = parse_num(row.pred_kp_uncertainty_top_left_xx_px2) + parse_num(row.pred_kp_uncertainty_top_left_yy_px2)
            cov_tr = parse_num(row.pred_kp_uncertainty_top_right_xx_px2) + parse_num(row.pred_kp_uncertainty_top_right_yy_px2)
            cov_br = parse_num(row.pred_kp_uncertainty_bottom_right_xx_px2) + parse_num(row.pred_kp_uncertainty_bottom_right_yy_px2)

            (
                gt_along = gt_along,
                gt_cross = gt_cross,
                gt_height = gt_height,
                est_along = est_along,
                est_cross = est_cross,
                est_height = est_height,
                err_along = est_along - gt_along,
                err_cross = est_cross - gt_cross,
                err_height = est_height - gt_height,
                integrity_passes = passes,
                p_value = integrity.p_value,
                pl_along = pl.alongtrack,
                pl_cross = pl.crosstrack,
                pl_alt = pl.altitude,
                gt_within_along = gt_within_along,
                gt_within_cross = gt_within_cross,
                gt_within_alt = gt_within_alt,
                cov_bottom_left = cov_bl,
                cov_top_left = cov_tl,
                cov_top_right = cov_tr,
                cov_bottom_right = cov_br,
            )
        catch e
            nothing  # Skip rows that fail
        end
    end

    filter(!isnothing, results)
end

# ╔═╡ a1b2c3d4-0023-0000-0000-000000000004
# Process trajectory with threading
trajectory_results = let
    use_cov = noise_source == "covariance"
    sigma = noise_source == "confidence" ? global_sigma : 5.0
    process_trajectory_threaded(df_good[1:min(1500, nrow(df_good)), :];
        use_covariance=use_cov, global_sigma=sigma, alpha=alpha_val)
end

# ╔═╡ a1b2c3d4-0023-0000-0000-000000000005
md"""
### Error Plots

Showing alongtrack, crosstrack, and altitude errors as we approach the runway.
The x-axis represents frame index (time progression toward runway).
"""

# ╔═╡ a1b2c3d4-0023-0000-0000-000000000006
# Create stacked error plots
let
    n = length(trajectory_results)
    indices = 1:n
    err_along = [r.err_along for r in trajectory_results]
    err_cross = [r.err_cross for r in trajectory_results]
    err_height = [r.err_height for r in trajectory_results]
    gt_along = [r.gt_along for r in trajectory_results]

    fig = Figure(size=(800, 600))

    ax1 = Axis(fig[1, 1], ylabel="Along-track error [m]", title="Pose Estimation Errors vs Ground Truth ($(selected_flight))")
    lines!(ax1, indices, err_along, color=:blue)
    hlines!(ax1, [0], color=:black, linestyle=:dash)

    ax2 = Axis(fig[2, 1], ylabel="Cross-track error [m]")
    lines!(ax2, indices, err_cross, color=:orange)
    hlines!(ax2, [0], color=:black, linestyle=:dash)

    ax3 = Axis(fig[3, 1], xlabel="Frame index", ylabel="Altitude error [m]")
    lines!(ax3, indices, err_height, color=:green)
    hlines!(ax3, [0], color=:black, linestyle=:dash)

    linkxaxes!(ax1, ax2, ax3)
    hidexdecorations!(ax1, grid=false)
    hidexdecorations!(ax2, grid=false)

    fig
end

# ╔═╡ a1b2c3d4-0025-0000-0000-000000000001
md"""
## Integrity Monitoring Plots

These plots show the protection level bounds computed from worst-case fault analysis.
- **Shaded band**: Protection level bounds `[estimate - PL, estimate + PL]`
- **Line**: Estimated value
- **Dots**: Ground truth (to validate if within bounds)
- **Color**: Green = integrity passes, Red = integrity fails
"""

# ╔═╡ a1b2c3d4-0025-0000-0000-000000000002
# Integrity statistics summary
let
    n = length(trajectory_results)
    n_pass = count(r -> r.integrity_passes, trajectory_results)
    n_gt_within_along = count(r -> r.gt_within_along, trajectory_results)
    n_gt_within_cross = count(r -> r.gt_within_cross, trajectory_results)
    n_gt_within_alt = count(r -> r.gt_within_alt, trajectory_results)

    println("=== Integrity Summary ($(selected_flight), α=$(alpha_val)) ===")
    println("Total frames: $n")
    println("Integrity passes: $n_pass ($(round(100*n_pass/n, digits=1))%)")
    println()
    println("=== GT Within Protection Levels ===")
    println("Along-track: $n_gt_within_along ($(round(100*n_gt_within_along/n, digits=1))%)")
    println("Cross-track: $n_gt_within_cross ($(round(100*n_gt_within_cross/n, digits=1))%)")
    println("Altitude:    $n_gt_within_alt ($(round(100*n_gt_within_alt/n, digits=1))%)")
end

# ╔═╡ a1b2c3d4-0025-0000-0000-000000000003
# Integrity monitoring plots with protection levels and corner covariances
let
    n = length(trajectory_results)
    indices = 1:n

    est_along = [r.est_along for r in trajectory_results]
    est_cross = [r.est_cross for r in trajectory_results]
    est_height = [r.est_height for r in trajectory_results]

    gt_along = [r.gt_along for r in trajectory_results]
    gt_cross = [r.gt_cross for r in trajectory_results]
    gt_height = [r.gt_height for r in trajectory_results]

    pl_along = [r.pl_along for r in trajectory_results]
    pl_cross = [r.pl_cross for r in trajectory_results]
    pl_alt = [r.pl_alt for r in trajectory_results]

    # Corner covariances (trace)
    cov_bl = [r.cov_bottom_left for r in trajectory_results]
    cov_tl = [r.cov_top_left for r in trajectory_results]
    cov_tr = [r.cov_top_right for r in trajectory_results]
    cov_br = [r.cov_bottom_right for r in trajectory_results]

    passes = [r.integrity_passes for r in trajectory_results]
    colors = [p ? :green : :red for p in passes]

    fig = Figure(size=(900, 1000))

    # Along-track plot
    ax1 = Axis(fig[1, 1], ylabel="Along-track [m]", title="Integrity Monitoring: $(selected_flight) (α=$(alpha_val))")
    band!(ax1, indices, est_along .- pl_along, est_along .+ pl_along, color=(:gray, 0.3))
    lines!(ax1, indices, est_along, color=:blue, linewidth=1, label="Estimate")
    scatter!(ax1, indices, gt_along, color=colors, markersize=4, label="GT")

    # Cross-track plot
    ax2 = Axis(fig[2, 1], ylabel="Cross-track [m]", limits=((nothing, nothing), (-100, 100)), yminorgridvisible=true)
    band!(ax2, indices, est_cross .- pl_cross, est_cross .+ pl_cross, color=(:gray, 0.3))
    lines!(ax2, indices, est_cross, color=:blue, linewidth=1)
    scatter!(ax2, indices, gt_cross, color=colors, markersize=4)

    # Altitude plot
    ax3 = Axis(fig[3, 1], ylabel="Altitude [m]", yminorgridvisible=true, yticks=0:50:500, limits=((nothing, nothing), (0, 500)))
    band!(ax3, indices, est_height .- pl_alt, est_height .+ pl_alt, color=(:gray, 0.3))
    lines!(ax3, indices, est_height, color=:blue, linewidth=1)
    scatter!(ax3, indices, gt_height, color=colors, markersize=4)

    # Corner covariance plot
    #ax4 = Axis(fig[4, 1], xlabel="Frame index", ylabel="Cov trace [px²]",
              # title="Keypoint Uncertainty (covariance trace)")
    #lines!(ax4, indices, cov_bl, color=:purple, linewidth=1, label="Near-left")
    #lines!(ax4, indices, cov_tl, color=:orange, linewidth=1, label="Far-left")
    #lines!(ax4, indices, cov_tr, color=:cyan, linewidth=1, label="Far-right")
    #lines!(ax4, indices, cov_br, color=:magenta, linewidth=1, label="Near-right")

    linkxaxes!(ax1, ax2, ax3)
    hidexdecorations!(ax1, grid=false)
    hidexdecorations!(ax2, grid=false)
    hidexdecorations!(ax3, grid=false)

    # Legend
    #Legend(fig[1, 2], ax1, framevisible=false)
    #Label(fig[2, 2], "Green = passes\nRed = fails", fontsize=10)
    #Legend(fig[4, 2], ax4, framevisible=false)

    fig
end

# ╔═╡ a1b2c3d4-0025-0000-0000-000000000004
md"""
### Errors vs Protection Levels

This plot shows the error magnitude compared to the protection level.
The GT is correctly bounded when `|error| < protection_level`.
"""

# ╔═╡ a1b2c3d4-0025-0000-0000-000000000005
# Plot errors relative to protection levels
let
    n = length(trajectory_results)
    indices = 1:n

    err_along = [abs(r.err_along) for r in trajectory_results]
    err_cross = [abs(r.err_cross) for r in trajectory_results]
    err_height = [abs(r.err_height) for r in trajectory_results]

    pl_along = [r.pl_along for r in trajectory_results]
    pl_cross = [r.pl_cross for r in trajectory_results]
    pl_alt = [r.pl_alt for r in trajectory_results]

    passes = [r.integrity_passes for r in trajectory_results]

    fig = Figure(size=(900, 700))

    ax1 = Axis(fig[1, 1], ylabel="|Error| [m]", title="Along-track: Error vs Protection Level ($(selected_flight))")
    lines!(ax1, indices, pl_along, color=:red, linewidth=1.5, label="Protection Level")
    lines!(ax1, indices, err_along, color=:blue, linewidth=1, label="|Error|")

    ax2 = Axis(fig[2, 1], ylabel="|Error| [m]", title="Cross-track")
    lines!(ax2, indices, pl_cross, color=:red, linewidth=1.5)
    lines!(ax2, indices, err_cross, color=:blue, linewidth=1)

    ax3 = Axis(fig[3, 1], xlabel="Frame index", ylabel="|Error| [m]", title="Altitude")
    lines!(ax3, indices, pl_alt, color=:red, linewidth=1.5)
    lines!(ax3, indices, err_height, color=:blue, linewidth=1)

    linkxaxes!(ax1, ax2, ax3)
    hidexdecorations!(ax1, grid=false)
    hidexdecorations!(ax2, grid=false)

    Legend(fig[1, 2], ax1, framevisible=false)

    fig
end

# ╔═╡ Cell order:
# ╠═3745f4fa-fbfa-11f0-8bbf-cd24cef3b17f
# ╠═a1b2c3d4-0001-0000-0000-000000000001
# ╟─a1b2c3d4-0002-0000-0000-000000000002
# ╠═a1b2c3d4-0003-0000-0000-000000000003
# ╠═a1b2c3d4-0004-0000-0000-000000000004
# ╠═510711e4-ad1b-4dc3-8a63-eca2873fd334
# ╠═7e52a693-c474-4631-af9c-c3c01767d90f
# ╠═ae66ebfc-5949-4119-b0c7-059a1bc5434a
# ╠═5d0d5c3a-0429-467d-a8d9-2778febb9ff2
# ╠═68d9df40-6870-4fa0-aa91-97bf15b839b8
# ╟─a1b2c3d4-0004-0001-0000-000000000001
# ╠═a1b2c3d4-0004-0002-0000-000000000001
# ╠═a1b2c3d4-0005-0000-0000-000000000005
# ╟─a1b2c3d4-0005-0001-0000-000000000001
# ╟─a1b2c3d4-0006-0000-0000-000000000006
# ╠═a1b2c3d4-0007-0000-0000-000000000007
# ╠═a1b2c3d4-0008-0000-0000-000000000008
# ╠═4abea7de-06f4-47db-9d5a-4e7b90b898d7
# ╟─aa29f607-c8fb-4ba7-8957-c75de18546de
# ╟─a1b2c3d4-0008-0001-0000-000000000001
# ╠═a1b2c3d4-0008-0002-0000-000000000001
# ╟─a1b2c3d4-0008-0003-0000-000000000003
# ╠═a1b2c3d4-0024-0000-0000-000000000001
# ╟─a1b2c3d4-0009-0000-0000-000000000009
# ╠═a1b2c3d4-0010-0000-0000-000000000010
# ╠═b2958518-1c72-4ed4-9a8b-cf2f271db480
# ╠═a1b2c3d4-0011-0000-0000-000000000011
# ╠═a1b2c3d4-0012-0000-0000-000000000012
# ╠═a1b2c3d4-0013-0000-0000-000000000013
# ╠═a1b2c3d4-0014-0000-0000-000000000014
# ╠═a1b2c3d4-0014-0001-0000-000000000001
# ╠═a1b2c3d4-0015-0000-0000-000000000015
# ╠═a081bc6a-1b1c-4bc3-83a2-36cfc1db5b2d
# ╠═a43924ff-7eab-4afd-a940-46333a38c35f
# ╟─a1b2c3d4-0016-0000-0000-000000000016
# ╠═a1b2c3d4-0017-0000-0000-000000000017
# ╠═a1b2c3d4-0018-0000-0000-000000000018
# ╠═a1b2c3d4-0019-0000-0000-000000000019
# ╠═a1b2c3d4-0020-0000-0000-000000000020
# ╟─a1b2c3d4-0021-0000-0000-000000000021
# ╠═a1b2c3d4-0022-0000-0000-000000000022
# ╠═dca746d9-5a4a-4386-830f-0bada1377367
# ╠═2e86ab6e-0949-11f1-a7b4-430f374d1f1a
# ╠═b6f1746a-0945-11f1-8bc5-d36f304c5851
# ╠═0c38d900-0947-11f1-bbb0-000000000001
# ╠═f640d1b4-ac63-4110-96d5-559287b43ae7
# ╟─0c38cd0a-0947-11f1-8be0-afa504b38a75
# ╟─0c38cf96-0947-11f1-90b5-43dc33093ab5
# ╠═0c38d460-0947-11f1-8a70-0f606135e992
# ╟─0c38d674-0947-11f1-990c-e7feb7e1db9f
# ╠═0c38d800-0947-11f1-aaa0-000000000001
# ╠═b66413ea-8f64-4016-a762-33b6795ecefa
# ╠═8789a9ec-0919-4b67-9261-98ba05100f87
# ╠═21a2f0ae-77ec-4190-828a-ba8aa4e3d7a5
# ╟─a1b2c3d4-0023-0000-0000-000000000001
# ╠═a1b2c3d4-0023-0000-0000-000000000002
# ╠═a1b2c3d4-0023-0000-0000-000000000003
# ╠═a1b2c3d4-0023-0000-0000-000000000004
# ╟─a1b2c3d4-0023-0000-0000-000000000005
# ╠═a1b2c3d4-0023-0000-0000-000000000006
# ╟─a1b2c3d4-0025-0000-0000-000000000001
# ╠═a1b2c3d4-0025-0000-0000-000000000002
# ╠═a1b2c3d4-0025-0000-0000-000000000003
# ╟─a1b2c3d4-0025-0000-0000-000000000004
# ╠═a1b2c3d4-0025-0000-0000-000000000005
