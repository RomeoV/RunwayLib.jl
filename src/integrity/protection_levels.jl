using Optimization: OptimizationFunction, OptimizationProblem, solve as opt_solve
using OptimizationOptimJL: NelderMead
using BracketingNonlinearSolve: IntervalNonlinearProblem, solve as bracket_solve
using Distributions: quantile

"""
    compute_zero_fault_protection_level(
        world_pts, observed_pts, noise_cov, cam_rot;
        alpha_idx=1, direction=1, prob=0.01, μ_schedule=[10.0, 1.0, 0.1], maxiters=10_000,
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
- `direction`: +1 to maximize, -1 to minimize position error
- `prob`: Probability for χ² bound (default 0.01)
- `μ_schedule`: Log-barrier penalty schedule (large=relaxed, small=tight)
- `maxiters`: Max iterations per Nelder-Mead stage

# Returns
NamedTuple with fields:
- `protection_level`: Absolute position error at solution (meters)
- `Δy`: Optimal whitened perturbation vector
- `pos_ref`: Baseline pose estimate
- `stat_ref`: Baseline integrity statistic
- `chi2_bound`: χ² quantile bound
- `feasible`: Whether both constraints are satisfied at solution
"""
function compute_zero_fault_protection_level(
    world_pts::AbstractVector{<:WorldPoint},
    observed_pts::AbstractVector{<:ProjectionPoint},
    noise_cov::AbstractMatrix,
    cam_rot::RotZYX;
    alpha_idx::Int = 1,
    direction::Int = 1,
    prob::Float64 = 0.01,
    μ_schedule = [10.0, 1.0, 0.1],
    maxiters::Int = 10_000,
)
    @assert alpha_idx in 1:3 "alpha_idx must be 1 (along-track), 2 (cross-track), or 3 (altitude)"
    @assert direction in (-1, 1) "direction must be +1 or -1"

    n_pts = length(world_pts)
    n_obs = 2 * n_pts

    # Baseline pose estimate (3-DOF, fixed rotation)
    pose_ref = estimatepose3dof(PointFeatures(world_pts, observed_pts), NO_LINES, cam_rot)
    pos_ref = pose_ref.pos
    pos_ref_stripped = ustrip.(m, pos_ref)
    pose_cache = pose_ref.cache  # reuse solver cache for all subsequent calls

    # Baseline integrity statistic
    stat_bl = compute_integrity_statistic(pos_ref, cam_rot, world_pts, observed_pts, noise_cov)
    stat_ref = stat_bl.stat
    dof = stat_bl.dofs
    chi2_bound = quantile(Chisq(dof), 1 - prob)

    # Flatten observations and extract noise scale
    obs_flat = reduce(vcat, SVector.(observed_pts))
    σ_val = sqrt(noise_cov[1, 1])px  # assume diagonal with equal variance

    # Tiny relaxation so the log-barrier is finite at Δy=0 (where stat = stat_ref exactly)
    stat_bound = stat_ref + 1e-4

    # --- Closures over the problem data ---
    function _eval_perturbation(Δy)
        # perturbed_flat = obs_flat + Δy * σ_val
        # perturbed_obs = observed_pts .+ [ProjectionPoint(el * px) for el in eachcol(reshape(perturbed_flat, 2, :))]
        perturbed_obs = observed_pts .+
            [
                ProjectionPoint(el...) * σ_val
                for el in eachcol(reshape(Δy, Size(2, length(observed_pts))))
            ]
        pose = estimatepose3dof(PointFeatures(world_pts, perturbed_obs), NO_LINES, cam_rot; cache=pose_cache)
        sr = compute_integrity_statistic(pose.pos, pose.rot, world_pts, perturbed_obs, noise_cov)
        return (; pose, sr, norm2=sum(Δy .^ 2))
    end

    function _penalized_objective(Δy, μ)
        ev = _eval_perturbation(Δy)
        obj = direction * -(ustrip(m, ev.pose.pos[alpha_idx]) - pos_ref_stripped[alpha_idx])
        slack_stat = stat_bound - ev.sr.stat
        slack_norm = chi2_bound - ev.norm2
        (slack_stat ≤ 0 || slack_norm ≤ 0) && return 1e10
        return obj + μ * (-log(slack_stat) - log(slack_norm))
    end

    # --- Progressive Nelder-Mead ---
    x0 = false.*similar(ustrip.(px, reduce(vcat, SVector.(observed_pts))))
    for μ in μ_schedule
        optf = OptimizationFunction(_penalized_objective)
        prob_opt = OptimizationProblem(optf, x0, μ)
        sol = opt_solve(prob_opt, NelderMead(); maxiters)
        successful_retcode(sol.retcode) || @warn "NM did not converge at μ=$μ: $(sol.retcode)"
        x0 = sol.u
    end

    # --- Line search to tighten constraints ---
    function _constraint_margin(t, Δy_dir)
        Δy = t * Δy_dir
        ev = _eval_perturbation(Δy)
        return max(ev.sr.stat - stat_ref, ev.norm2 - chi2_bound)
    end

    margin_at_1 = _constraint_margin(1.0, x0)
    if abs(margin_at_1) > 1e-8  # not already on boundary
        Δy_dir = copy(x0)
        if margin_at_1 < 0
            # Feasible — push outward
            t_max = 1.5
            while _constraint_margin(t_max, Δy_dir) < 0
                t_max *= 1.5
            end
            prob_bisect = IntervalNonlinearProblem(_constraint_margin, (1.0, t_max), Δy_dir)
        else
            # Infeasible — scale inward
            prob_bisect = IntervalNonlinearProblem(_constraint_margin, (1e-4, 1.0), Δy_dir)
        end
        t_sol = bracket_solve(prob_bisect)
        if successful_retcode(t_sol)
            x0 = t_sol.u * Δy_dir
        end
    end

    # --- Compute final result ---
    ev_final = _eval_perturbation(x0)
    pos_error = ustrip(m, ev_final.pose.pos[alpha_idx]) - pos_ref_stripped[alpha_idx]
    feasible = ev_final.sr.stat ≤ stat_ref + 1e-6 && ev_final.norm2 ≤ chi2_bound + 1e-6

    return (;
        protection_level = abs(pos_error),
        Δy = x0,
        pos_ref,
        stat_ref,
        stat = ev_final.sr.stat,
        norm2 = ev_final.norm2,
        chi2_bound,
        feasible,
    )
end
