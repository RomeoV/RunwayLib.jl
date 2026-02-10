using Optimization: OptimizationFunction, OptimizationProblem, solve as opt_solve
using OptimizationOptimJL: NelderMead
using BracketingNonlinearSolve: IntervalNonlinearProblem, solve as bracket_solve
using Distributions: quantile
using Printf: @sprintf

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

    # --- Shared closures ---
    n_solves = Ref(0)
    function _eval_perturbation(Δy)
        n_solves[] += 1
        perturbed_obs = observed_pts .+
            [
                ProjectionPoint(el...) * σ_val
                for el in eachcol(reshape(Δy, Size(2, length(observed_pts))))
            ]
        pose = estimatepose3dof(PointFeatures(world_pts, perturbed_obs), NO_LINES, cam_rot; cache=pose_cache)
        sr = compute_integrity_statistic(pose.pos, pose.rot, world_pts, perturbed_obs, noise_cov)
        return (; pose, sr, norm2=sum(Δy .^ 2))
    end

    function _constraint_margin(t, Δy_dir)
        ev = _eval_perturbation(t * Δy_dir)
        return max(ev.sr.stat - stat_ref, ev.norm2 - chi2_bound)
    end

    """Scale `Δy_dir` to the constraint boundary via bisection. Returns the projected vector."""
    function _project_to_boundary(Δy_dir)
        norm(Δy_dir) < 1e-12 && return Δy_dir
        margin = _constraint_margin(1.0, Δy_dir)
        abs(margin) ≤ 1e-8 && return Δy_dir  # already on boundary

        if margin < 0
            # Feasible — expand to find infeasible upper bound
            t_hi = 1.5
            while _constraint_margin(t_hi, Δy_dir) < 0
                t_hi *= 1.5
            end
            t_lo = 1.0
        else
            # Infeasible — halve to find a strictly feasible lower bound
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

    # --- Solve one direction: NM + boundary projection ---
    function _solve_direction(dir::Int, x0)
        # Project initial condition to boundary (ensures feasibility for warm-starts)
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

        # Push NM result to constraint boundary
        x0 = _project_to_boundary(x0)

        ev_final = _eval_perturbation(x0)
        pos_error = ustrip(m, ev_final.pose.pos[alpha_idx]) - pos_ref_stripped[alpha_idx]
        feasible = ev_final.sr.stat ≤ stat_ref + 1e-6 && ev_final.norm2 ≤ chi2_bound + 1e-6

        return (; protection_level=pos_error, Δy=x0,
                  stat=ev_final.sr.stat, norm2=ev_final.norm2, feasible)
    end

    x0 = false .* similar(ustrip.(px, reduce(vcat, SVector.(observed_pts))))
    shared = (; pos_ref, stat_ref, chi2_bound)
    result = _dispatch_direction(Val(direction), _solve_direction, x0, shared)
    verbose && _print_pl_summary(Val(direction), result, alpha_idx, n_solves[])
    return merge(result, (; n_solves=n_solves[]))
end

# Single direction
_dispatch_direction(::Val{D}, solve_fn, x0, shared) where {D} =
    merge(solve_fn(D, x0), shared)

# Both directions: solve lo first, warm-start hi with flipped Δy
_dispatch_direction(::Val{0}, solve_fn, x0, shared) = let
    lo = solve_fn(-1, x0)
    hi = solve_fn(1, -lo.Δy)
    merge((; lo, hi), shared)
end

const _ALPHA_LABELS = ("along-track", "cross-track", "altitude")

function _print_pl_row(io, label, r, stat_ref, chi2_bound)
    feas = r.feasible ? "yes" : "NO"
    row = @sprintf("  │ %3s │ %+10.3f │ %8.4f / %-8.4f │ %7.3f / %-7.3f │ %3s │",
        label, r.protection_level, r.stat, stat_ref, r.norm2, chi2_bound, feas)
    println(io, row)
end

function _print_pl_summary(::Val{0}, result, alpha_idx, n_solves)
    io = stdout
    header = "Zero-fault PL: α=$alpha_idx ($(_ALPHA_LABELS[alpha_idx]))"
    println(io, "  ┌─────┬────────────┬─────────────────────┬───────────────────┬─────┐")
    println(io, "  │ dir │     PL (m) │    stat / bound     │   norm² / bound   │ ok? │")
    println(io, "  ├─────┼────────────┼─────────────────────┼───────────────────┼─────┤")
    _print_pl_row(io, "lo", result.lo, result.stat_ref, result.chi2_bound)
    _print_pl_row(io, "hi", result.hi, result.stat_ref, result.chi2_bound)
    println(io, "  └─────┴────────────┴─────────────────────┴───────────────────┴─────┘")
    println(io, "  $header  ($n_solves pose solves)")
end

function _print_pl_summary(::Val{D}, result, alpha_idx, n_solves) where {D}
    io = stdout
    label = D == 1 ? "hi" : "lo"
    header = "Zero-fault PL: α=$alpha_idx ($(_ALPHA_LABELS[alpha_idx])), dir=$D"
    println(io, "  ┌─────┬────────────┬─────────────────────┬───────────────────┬─────┐")
    println(io, "  │ dir │     PL (m) │    stat / bound     │   norm² / bound   │ ok? │")
    println(io, "  ├─────┼────────────┼─────────────────────┼───────────────────┼─────┤")
    _print_pl_row(io, label, result, result.stat_ref, result.chi2_bound)
    println(io, "  └─────┴────────────┴─────────────────────┴───────────────────┴─────┘")
    println(io, "  $header  ($n_solves pose solves)")
end
