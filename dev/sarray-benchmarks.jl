using RunwayLib, Unitful.DefaultSymbols, Rotations
using StaticArrays
using Chairmarks

runway_corners = [
    WorldPoint(0.0m, 50m, 0m),     # near left
    WorldPoint(3000.0m, 50m, 0m),  # far left
    WorldPoint(3000.0m, -50m, 0m),  # far right
    WorldPoint(0.0m, -50m, 0m),    # near right
]

cam_pos = WorldPoint(-2000.0m, 12m, 150m)
cam_rot = RotZYX(roll=1.5°, pitch=5°, yaw=0°)

true_observations = [project(cam_pos, cam_rot, p) for p in runway_corners]
noisy_observations = [p + ProjectionPoint(2.0*randn(2)px) for p in true_observations]

pf_arr = PointFeatures(runway_corners, noisy_observations)
pf_sa = PointFeatures(SVector{4}(runway_corners), SVector{4}(noisy_observations))

## Start with 6dof

minimum(@be pf_arr estimatepose6dof)
minimum(@be pf_sa estimatepose6dof)

# with caches preallocated
let cache = estimatepose6dof(pf_arr).cache
    minimum(@be pf_arr estimatepose6dof(_; cache))
end

let cache = estimatepose6dof(pf_sa).cache
    minimum(@be pf_sa estimatepose6dof(_; cache))
end

## Now 3dof

minimum(@be pf_arr estimatepose3dof(_, NO_LINES, cam_rot))
minimum(@be pf_sa estimatepose3dof(_, NO_LINES, cam_rot))

# with caches preallocated
let cache = estimatepose3dof(pf_arr, NO_LINES, cam_rot).cache
    minimum(@be pf_arr estimatepose3dof(_, NO_LINES, cam_rot; cache))
end

let cache = estimatepose3dof(pf_sa, NO_LINES, cam_rot).cache
    minimum(@be pf_sa estimatepose3dof(_, NO_LINES, cam_rot; cache))
end


## Including lines
line_pts = [
    (runway_corners[1], runway_corners[2]),
    (runway_corners[3], runway_corners[4]),
]
true_lines = map(line_pts) do (p1, p2)
    proj1 = project(cam_pos, cam_rot, p1)
    proj2 = project(cam_pos, cam_rot, p2)
    getline(proj1, proj2)
end
observed_lines = [
  Line(
    r + 1px*randn(),
    theta + deg2rad(1°)*randn()
  )
  for (; r, theta) in true_lines
]
lf_arr = LineFeatures(line_pts, observed_lines)
lf_sa = LineFeatures(SVector{2}(line_pts), SVector{2}(observed_lines))

minimum(@be (pf_arr, lf_arr) estimatepose6dof(_...))
minimum(@be (pf_sa, lf_sa) estimatepose6dof(_...))

# with caches preallocated
let cache = estimatepose6dof(pf_arr, lf_arr).cache
    minimum(@be (pf_arr, lf_arr) estimatepose6dof(_...; cache=cache))
end

let cache = estimatepose6dof(pf_sa, lf_sa).cache
    minimum(@be (pf_sa, lf_sa) estimatepose6dof(_...; cache))
end

## Including lines w/ 3dof

minimum(@be (pf_arr, lf_arr) estimatepose3dof(_..., cam_rot))
minimum(@be (pf_sa, lf_sa) estimatepose3dof(_..., cam_rot))

# with caches preallocated
let cache = estimatepose3dof(pf_arr, lf_arr, cam_rot).cache
    minimum(@be (pf_arr, lf_arr) estimatepose3dof(_..., cam_rot; cache))
end

let cache = estimatepose3dof(pf_sa, lf_sa, cam_rot).cache
    minimum(@be (pf_sa, lf_sa) estimatepose3dof(_..., cam_rot; cache))
end
