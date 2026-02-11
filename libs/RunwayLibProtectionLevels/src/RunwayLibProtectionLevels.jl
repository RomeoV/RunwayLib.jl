"""
Gradient-based zero-fault protection level computation using Ipopt.

Uses a bilevel optimization: NonlinearSolve (inner pose solver with IFT) +
Ipopt (outer constrained optimizer). ~30x faster than the derivative-free
NelderMead approach in RunwayLib.
"""
module RunwayLibProtectionLevels

using RunwayLib
using Unitful, Unitful.DefaultSymbols
using StaticArrays
using LinearAlgebra
using Distributions: quantile, Chisq
using Rotations: RotZYX
using Printf: @sprintf

using SciMLBase: NonlinearLeastSquaresProblem, solve as nls_solve
using NonlinearSolveFirstOrder: LevenbergMarquardt

using Optimization: OptimizationFunction, OptimizationProblem, solve as opt_solve
using OptimizationIpopt
using ADTypes: AutoForwardDiff

import RunwayLib: compute_whitened_parity_residual, PointFeatures, px, NO_LINES,
    CAMERA_CONFIG_OFFSET, estimatepose3dof

export compute_zero_fault_protection_level_grad

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
    pl = dir * (pos_sol[alpha_idx] - pos_ref_m[alpha_idx])
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

end # module
