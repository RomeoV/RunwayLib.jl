"""
Protection level computation for runway pose estimation.

Provides both gradient-based (Ipopt, ~30x faster) and derivative-free (NelderMead)
zero-fault protection level solvers, plus worst-case fault direction analysis.
"""
module RunwayLibProtectionLevels

using RunwayLib
using Unitful, Unitful.DefaultSymbols
using StaticArrays
using SparseArrays
using LinearAlgebra
using Distributions: quantile, Chisq
using Rotations: RotZYX
using Printf: @sprintf

using SciMLBase: NonlinearLeastSquaresProblem, solve as nls_solve, successful_retcode
using NonlinearSolveFirstOrder: LevenbergMarquardt
using BracketingNonlinearSolve: IntervalNonlinearProblem, solve as bracket_solve

using Optimization: OptimizationFunction, OptimizationProblem, solve as opt_solve
using OptimizationOptimJL: NelderMead
using OptimizationIpopt
using ADTypes: AutoForwardDiff

import RunwayLib: compute_whitened_parity_residual, compute_integrity_statistic,
    PointFeatures, px, NO_LINES, CAMERA_CONFIG_OFFSET, estimatepose3dof

export compute_zero_fault_protection_level_grad
export compute_zero_fault_protection_level
export compute_worst_case_fault_direction_and_slope

# =============================================================================
# Helpers
# =============================================================================

_optvar2nominal_3dof(x) = SA[-exp(x[1]), x[2], exp(x[3])]
_nominal2optvar_3dof(x) = SA[log(-x[1]), x[2], log(x[3])]
_pos_to_world(pos_m) = WorldPoint(pos_m .* m)

function _obs_to_points(obs, ::Val{N}) where {N}
    ntuple(i -> ProjectionPoint(obs[2i-1] * px, obs[2i] * px), Val(N))
end

function _obs_to_flat(observed_pts)
    reduce(vcat, [SVector(ustrip(px, p.x), ustrip(px, p.y)) for p in observed_pts])
end

# =============================================================================
# NLS residual (obs as parameter → triggers NonlinearSolve IFT when Dual-valued)
# =============================================================================

function _make_pose_residuals(cam_rot, corners, Linv, ::Val{N}) where {N}
    function (optvar, obs::AbstractVector)
        pos_m = _optvar2nominal_3dof(optvar)
        cam_pos = _pos_to_world(pos_m)
        obs_pts = _obs_to_points(obs, Val(N))
        projected = ntuple(i -> project(cam_pos, cam_rot, corners[i]), Val(N))
        errors = reduce(vcat, ntuple(i -> SVector(projected[i] - obs_pts[i]), Val(N)))
        weighted = (Linv * errors) ./ px
        return ustrip.(NoUnits, weighted)
    end
end

# =============================================================================
# Pose solver dispatch: Float64 (cached) vs Dual (allocating + IFT)
# =============================================================================

function _solve_pose(obs::Vector{Float64}, corners, cam_rot, pose_cache, ::Val{N}) where {N}
    obs_pts = SVector(_obs_to_points(obs, Val(N))...)
    pf_new = PointFeatures(corners, obs_pts)
    try
        pose = estimatepose3dof(pf_new, NO_LINES, cam_rot; cache=pose_cache)
        return SVector(ustrip.(m, pose.pos)...)
    catch
        return SVector(NaN, NaN, NaN)
    end
end

function _solve_pose(obs::AbstractVector, residual_fn, u0)
    prob = NonlinearLeastSquaresProblem(residual_fn, u0, obs)
    sol = nls_solve(prob, LevenbergMarquardt())
    return _optvar2nominal_3dof(sol.u)
end

# =============================================================================
# Integrity stat via RunwayLib (AD-safe after pinv→LU fix)
# =============================================================================

function _compute_stat(pos_m, obs, corners, cam_rot, cov, Linv, ::Val{N}) where {N}
    cam_pos = _pos_to_world(pos_m)
    obs_pts = collect(_obs_to_points(obs, Val(N)))
    pf = PointFeatures(corners, obs_pts, CAMERA_CONFIG_OFFSET, cov, Linv)
    r = compute_whitened_parity_residual(cam_pos, cam_rot, pf)
    return dot(r, r)
end

# =============================================================================
# Direction solver (Ipopt with proper constraint enforcement)
# =============================================================================

function _solve_direction(
    dir, alpha_idx,
    obs_flat, σ_val, pos_ref_m,
    corners, cam_rot, pose_cache, residual_fn, u0_opt,
    cov, Linv, stat_bound, chi2_bound, ::Val{N},
) where {N}
    n_obs = 2N

    function objective(Δy, _)
        obs_perturbed = collect(obs_flat) + σ_val * Δy
        pos = _solve_pose(obs_perturbed, residual_fn, u0_opt)
        return -dir * (pos[alpha_idx] - pos_ref_m[alpha_idx])
    end

    function constraints!(res, Δy, _)
        obs_perturbed = collect(obs_flat) + σ_val * Δy
        pos = _solve_pose(obs_perturbed, residual_fn, u0_opt)
        res[1] = _compute_stat(pos, obs_perturbed, corners, cam_rot, cov, Linv, Val(N))
        res[2] = sum(Δy .^ 2)
        return nothing
    end

    optf = OptimizationFunction(objective, AutoForwardDiff(); cons=constraints!)
    prob = OptimizationProblem(optf, zeros(n_obs), nothing;
        lcons=[-Inf, -Inf], ucons=[stat_bound, chi2_bound])
    sol = opt_solve(prob, IpoptOptimizer(); print_level=0)

    # Evaluate solution at Float64 for reporting
    obs_sol = collect(obs_flat) + σ_val * sol.u
    pos_sol = _solve_pose(obs_sol, corners, cam_rot, pose_cache, Val(N))
    stat_sol = _compute_stat(pos_sol, obs_sol, corners, cam_rot, cov, Linv, Val(N))
    norm2_sol = sum(sol.u .^ 2)
    pl = pos_sol[alpha_idx] - pos_ref_m[alpha_idx]
    feasible = stat_sol ≤ stat_bound + 1e-6 && norm2_sol ≤ chi2_bound + 1e-4

    return (; protection_level=pl, Δy=sol.u, stat=stat_sol, norm2=norm2_sol, feasible)
end

# =============================================================================
# Direction dispatch (same pattern as RunwayLib's NM version)
# =============================================================================

_dispatch_direction(::Val{D}, solve_fn, shared) where {D} =
    merge(solve_fn(D), shared)

function _dispatch_direction(::Val{0}, solve_fn, shared)
    lo = solve_fn(-1)
    hi = solve_fn(1)
    merge((; lo, hi), shared)
end

# =============================================================================
# Main API
# =============================================================================

"""
    compute_zero_fault_protection_level_grad(
        world_pts, observed_pts, noise_cov, cam_rot;
        alpha_idx=1, direction=0, prob=0.01, verbose=false,
    )

Compute the zero-fault protection level using gradient-based optimization (Ipopt).

Same interface and return shape as `RunwayLib.compute_zero_fault_protection_level`,
but ~30x faster by using Ipopt (interior-point with exact gradients via ForwardDiff +
implicit function theorem through the pose solver).

See `RunwayLib.compute_zero_fault_protection_level` for full documentation of arguments.
"""
function compute_zero_fault_protection_level_grad(
    world_pts::AbstractVector{<:WorldPoint},
    observed_pts::AbstractVector{<:ProjectionPoint},
    noise_cov::AbstractMatrix,
    cam_rot::RotZYX;
    alpha_idx::Int = 1,
    direction::Int = 0,
    prob::Float64 = 0.01,
    verbose::Bool = false,
)
    @assert alpha_idx in 1:3 "alpha_idx must be 1 (along-track), 2 (cross-track), or 3 (altitude)"
    @assert direction in (-1, 0, 1) "direction must be 0 (both), +1, or -1"

    n_pts = length(world_pts)
    corners = SVector{n_pts}(world_pts)

    # Baseline pose (3-DOF, fixed rotation)
    pf_ref = PointFeatures(corners, SVector{n_pts}(observed_pts))
    pose_ref = estimatepose3dof(pf_ref, NO_LINES, cam_rot)
    pos_ref_m = SVector(ustrip.(m, pose_ref.pos)...)

    # Baseline integrity statistic
    stat_bl = compute_integrity_statistic(pose_ref.pos, cam_rot, world_pts, observed_pts, noise_cov)
    stat_ref = stat_bl.stat
    chi2_bound = quantile(Chisq(stat_bl.dofs), 1 - prob)
    stat_bound = stat_ref + 1e-4  # tiny relaxation for feasibility at Δy=0

    # Precompute scenario data
    obs_flat = _obs_to_flat(observed_pts)
    u0_opt = collect(_nominal2optvar_3dof(pos_ref_m))
    σ_val = sqrt(noise_cov[1, 1])  # assumes diagonal with equal variance
    cov = pf_ref.cov
    Linv = pf_ref.Linv
    pose_cache = pose_ref.cache
    residual_fn = _make_pose_residuals(cam_rot, corners, Linv, Val(n_pts))

    shared = (; pos_ref=pose_ref.pos, stat_ref, chi2_bound)

    solve_fn = dir -> _solve_direction(
        dir, alpha_idx,
        obs_flat, σ_val, pos_ref_m,
        corners, cam_rot, pose_cache, residual_fn, u0_opt,
        cov, Linv, stat_bound, chi2_bound, Val(n_pts),
    )

    result = _dispatch_direction(Val(direction), solve_fn, shared)

    if verbose
        _print_summary(Val(direction), result, alpha_idx)
    end

    return result
end

# =============================================================================
# Verbose output
# =============================================================================

const _ALPHA_LABELS = ("along-track", "cross-track", "altitude")

function _print_row(io, label, r, stat_ref, chi2_bound)
    feas = r.feasible ? "yes" : "NO"
    row = @sprintf("  │ %3s │ %+10.3f │ %8.4f / %-8.4f │ %7.3f / %-7.3f │ %3s │",
        label, r.protection_level, r.stat, stat_ref, r.norm2, chi2_bound, feas)
    println(io, row)
end

function _print_summary(::Val{0}, result, alpha_idx)
    io = stdout
    println(io, "  ┌─────┬────────────┬─────────────────────┬───────────────────┬─────┐")
    println(io, "  │ dir │     PL (m) │    stat / bound     │   norm² / bound   │ ok? │")
    println(io, "  ├─────┼────────────┼─────────────────────┼───────────────────┼─────┤")
    _print_row(io, "lo", result.lo, result.stat_ref, result.chi2_bound)
    _print_row(io, "hi", result.hi, result.stat_ref, result.chi2_bound)
    println(io, "  └─────┴────────────┴─────────────────────┴───────────────────┴─────┘")
    println(io, "  Zero-fault PL (Ipopt): α=$alpha_idx ($(_ALPHA_LABELS[alpha_idx]))")
end

function _print_summary(::Val{D}, result, alpha_idx) where {D}
    io = stdout
    label = D == 1 ? "hi" : "lo"
    println(io, "  ┌─────┬────────────┬─────────────────────┬───────────────────┬─────┐")
    println(io, "  │ dir │     PL (m) │    stat / bound     │   norm² / bound   │ ok? │")
    println(io, "  ├─────┼────────────┼─────────────────────┼───────────────────┼─────┤")
    _print_row(io, label, result, result.stat_ref, result.chi2_bound)
    println(io, "  └─────┴────────────┴─────────────────────┴───────────────────┴─────┘")
    println(io, "  Zero-fault PL (Ipopt): α=$alpha_idx ($(_ALPHA_LABELS[alpha_idx])), dir=$D")
end

# =============================================================================
# NelderMead-based protection level (derivative-free, slower but robust)
# =============================================================================

"""
    compute_zero_fault_protection_level(
        world_pts, observed_pts, noise_cov, cam_rot;
        alpha_idx=1, direction=0, prob=0.01, μ_schedule=[10.0, 1.0, 0.1], maxiters=10_000,
    )

Compute the zero-fault hypothesis protection level via constrained optimization.

Finds the worst-case observation perturbation `Δy` (in whitened coordinates) that
maximizes (or minimizes) position error along axis `alpha_idx`, subject to:
1. The integrity statistic does not increase beyond the baseline value
2. The perturbation norm stays within the χ² bound at probability `prob`

Uses Nelder-Mead with log-barrier penalties and progressive tightening, followed by
a line search to push the solution to the constraint boundary.

# Arguments
- `world_pts`: Known 3D feature locations (runway corners)
- `observed_pts`: Observed 2D projections (with noise)
- `noise_cov`: Measurement noise covariance matrix (in pixels²)
- `cam_rot`: Known camera attitude (3-DOF mode)
- `alpha_idx`: Position component (1=along-track, 2=cross-track, 3=altitude)
- `direction`: 0 for both directions (default), +1 to maximize, -1 to minimize
- `prob`: Probability for χ² bound (default 0.01)
- `μ_schedule`: Log-barrier penalty schedule (large=relaxed, small=tight)
- `maxiters`: Max iterations per Nelder-Mead stage

# Returns
For `direction=0`: NamedTuple `(; lo, hi, pos_ref, stat_ref, chi2_bound)` where
`lo` and `hi` each contain `(; protection_level, Δy, stat, norm2, feasible)`.

For `direction=±1`: flat NamedTuple with all fields.
"""
function compute_zero_fault_protection_level(
    world_pts::AbstractVector{<:WorldPoint},
    observed_pts::AbstractVector{<:ProjectionPoint},
    noise_cov::AbstractMatrix,
    cam_rot::RotZYX;
    alpha_idx::Int = 1,
    direction::Int = 0,
    prob::Float64 = 0.01,
    μ_schedule = [10.0, 1.0, 0.1],
    maxiters::Int = 10_000,
    verbose::Bool = false,
)
    @assert alpha_idx in 1:3 "alpha_idx must be 1 (along-track), 2 (cross-track), or 3 (altitude)"
    @assert direction in (-1, 0, 1) "direction must be 0 (both), +1, or -1"

    # Baseline pose estimate (3-DOF, fixed rotation)
    pose_ref = estimatepose3dof(PointFeatures(world_pts, observed_pts), NO_LINES, cam_rot)
    pos_ref = pose_ref.pos
    pos_ref_stripped = ustrip.(m, pos_ref)
    pose_cache = pose_ref.cache

    # Baseline integrity statistic
    stat_bl = compute_integrity_statistic(pos_ref, cam_rot, world_pts, observed_pts, noise_cov)
    stat_ref = stat_bl.stat
    dof = stat_bl.dofs
    chi2_bound = quantile(Chisq(dof), 1 - prob)

    # Noise scale (assume diagonal with equal variance)
    σ_val = sqrt(noise_cov[1, 1])px

    # Tiny relaxation so the log-barrier is finite at Δy=0
    stat_bound = stat_ref + 1e-4

    n_solves = Ref(0)
    function _eval_perturbation(Δy)
        n_solves[] += 1
        perturbed_obs = observed_pts .+
            [
                ProjectionPoint(el...) * σ_val
                for el in eachcol(reshape(Δy, 2, length(observed_pts)))
            ]
        pose = estimatepose3dof(PointFeatures(world_pts, perturbed_obs), NO_LINES, cam_rot; cache=pose_cache)
        sr = compute_integrity_statistic(pose.pos, pose.rot, world_pts, perturbed_obs, noise_cov)
        return (; pose, sr, norm2=sum(Δy .^ 2))
    end

    function _constraint_margin(t, Δy_dir)
        ev = _eval_perturbation(t * Δy_dir)
        return max(ev.sr.stat - stat_ref, ev.norm2 - chi2_bound)
    end

    function _project_to_boundary(Δy_dir)
        norm(Δy_dir) < 1e-12 && return Δy_dir
        margin = _constraint_margin(1.0, Δy_dir)
        abs(margin) ≤ 1e-8 && return Δy_dir

        if margin < 0
            t_hi = 1.5
            while _constraint_margin(t_hi, Δy_dir) < 0
                t_hi *= 1.5
            end
            t_lo = 1.0
        else
            t_lo, t_hi = 0.5, 1.0
            for _ in 1:50
                _constraint_margin(t_lo, Δy_dir) < 0 && break
                t_lo *= 0.5
            end
            _constraint_margin(t_lo, Δy_dir) ≥ 0 && return t_lo * Δy_dir
        end
        t_sol = bracket_solve(IntervalNonlinearProblem(_constraint_margin, (t_lo, t_hi), Δy_dir))
        return successful_retcode(t_sol) ? t_sol.u * Δy_dir : Δy_dir
    end

    function _solve_direction_nm(dir::Int, x0)
        x0 = _project_to_boundary(x0)

        function _penalized_objective(Δy, μ)
            ev = _eval_perturbation(Δy)
            obj = dir * -(ustrip(m, ev.pose.pos[alpha_idx]) - pos_ref_stripped[alpha_idx])
            slack_stat = stat_bound - ev.sr.stat
            slack_norm = chi2_bound - ev.norm2
            (slack_stat ≤ 0 || slack_norm ≤ 0) && return 1e10
            return obj + μ * (-log(slack_stat) - log(slack_norm))
        end

        for μ in μ_schedule
            optf = OptimizationFunction(_penalized_objective)
            prob_opt = OptimizationProblem(optf, x0, μ)
            sol = opt_solve(prob_opt, NelderMead(); maxiters)
            successful_retcode(sol.retcode) || @warn "NM did not converge at μ=$μ: $(sol.retcode)"
            x0 = sol.u
        end

        x0 = _project_to_boundary(x0)

        ev_final = _eval_perturbation(x0)
        pos_error = ustrip(m, ev_final.pose.pos[alpha_idx]) - pos_ref_stripped[alpha_idx]
        feasible = ev_final.sr.stat ≤ stat_ref + 1e-6 && ev_final.norm2 ≤ chi2_bound + 1e-6

        return (; protection_level=pos_error, Δy=x0,
                  stat=ev_final.sr.stat, norm2=ev_final.norm2, feasible)
    end

    x0 = false .* similar(ustrip.(px, reduce(vcat, SVector.(observed_pts))))
    shared = (; pos_ref, stat_ref, chi2_bound)

    _nm_dispatch(::Val{D}, x0) where {D} = merge(_solve_direction_nm(D, x0), shared)
    function _nm_dispatch(::Val{0}, x0)
        lo = _solve_direction_nm(-1, x0)
        hi = _solve_direction_nm(1, -lo.Δy)
        merge((; lo, hi), shared)
    end

    result = _nm_dispatch(Val(direction), x0)
    verbose && _print_summary(Val(direction), result, alpha_idx)
    return merge(result, (; n_solves=n_solves[]))
end

# =============================================================================
# Worst-case fault direction and slope
# =============================================================================

"""
    compute_worst_case_fault_direction_and_slope(
        alpha_idx::Int,
        fault_indices::AbstractVector{Int},
        H::AbstractMatrix,
        noise_cov::AbstractMatrix,
    )

Computes the worst-case fault direction and corresponding failure mode slope
for a selected pose parameter and fault subset.

# Arguments
- `alpha_idx::Int`: Monitored parameter index
  - 1 = along-track position
  - 2 = cross-track position
  - 3 = height above runway
  - 4 = yaw
  - 5 = pitch
  - 6 = roll
- `fault_indices::AbstractVector{Int}`: Indices of measurements in fault subset
- `H::AbstractMatrix`: Jacobian matrix (ndof columns)
- `noise_cov::AbstractMatrix`: Measurement noise covariance matrix

# Returns
- `f_dir`: Worst-case fault direction (normalized vector)
- `g_slope`: Failure mode slope (quantifies sensitivity to faults in this direction)

!!! note
    The Jacobian `H` must have the correct number of columns for the degrees of
    freedom (3 for position-only, 6 for full pose estimation).
"""
function compute_worst_case_fault_direction_and_slope(
    alpha_idx::Int,
    fault_indices::AbstractVector{Int},
    H::AbstractMatrix,
    noise_cov::AbstractMatrix,
)
    @assert 1 <= alpha_idx <= size(H, 2) "alpha_idx must be 1-ndof"
    @assert all(1 .<= fault_indices .<= size(H, 1)) "fault_indices must in `1:size(H, 1)`"

    ndof = size(H, 2)
    α = SVector(ntuple(i -> i == alpha_idx ? 1.0 : 0.0, Val(ndof)))

    S_0 = (H' * H) \ H'
    s_0 = S_0' * α

    L, _ = cholesky(noise_cov)
    Linv = inv(L)

    proj_parity = I - H * S_0
    proj_parity_Linv = proj_parity * Linv

    n_measurements = size(H, 1)
    n_faults = length(fault_indices)
    A_i = (
        sparse(collect(fault_indices), 1:n_faults, ones(n_faults), n_measurements, n_faults)
    ) |> SMatrix{n_measurements, n_faults}

    visibility_matrix = A_i' * proj_parity_Linv * proj_parity_Linv' * A_i
    m_Xi = A_i' * s_0

    f_dir = A_i * (visibility_matrix \ m_Xi) |> normalize

    g_slope_squared = m_Xi' * (visibility_matrix \ m_Xi)
    @assert g_slope_squared >= 0 "Computed negative slope squared, numerical issue?"
    g_slope = sqrt(g_slope_squared)

    return f_dir, g_slope
end

end # module
