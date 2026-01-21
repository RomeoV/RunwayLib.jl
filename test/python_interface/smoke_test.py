import numpy as np
from juliacall import Main as jl
jl.seval("using RunwayLib, PythonCall")
jl.seval("import RunwayLib: px, m")

points2d = [
    jl.ProjectionPoint(2284.8, 619.32)*jl.px,
    jl.ProjectionPoint(2355.2, 620.90)*jl.px,
    jl.ProjectionPoint(2399.5, 903.30)*jl.px,
    jl.ProjectionPoint(2252.4, 904.09)*jl.px,
]
points3d = [
    jl.WorldPoint(1600, 15, -8)*jl.m,
    jl.WorldPoint(1600, -15, -8)*jl.m,
    jl.WorldPoint(0, -15, 0)*jl.m,
    jl.WorldPoint(0, 15, 0)*jl.m,
]
intrinsic_matrix = np.array(
    [
        [-7311.8, 0.0, 2032.3],
        [0.0, -7311.8, 1707.4],
        [0.0, 0.0, 1.0],
    ]
)

camconf = jl.CameraMatrix[jl.Symbol("offset")](
    intrinsic_matrix, jl.px(2048.0), jl.px(1024.0),
)

jl.seval("using RunwayLib.Unitful: ustrip, rad")

line_pts = [
    (points3d[1], points3d[2]),  # near left, far left, according to points3d
    (points3d[0], points3d[3])   # near right, far right, according to points3d
];
observed_lines = [
    # these are bogus numbers for illustration purposes only
    jl.Line(
        jl.px(2000.0),
        jl.rad(np.deg2rad(1))
    ),
    jl.Line(
        jl.px(2000.0),
        jl.rad(np.deg2rad(-1))
    )
];

res_jl = jl.estimatepose6dof(
    jl.PointFeatures(points3d, points2d, camconf),
    jl.LineFeatures(line_pts, observed_lines, camconf)
)
pos = np.asarray(jl.broadcast(jl.ustrip, res_jl.pos))  # or `np.array(..., copy=None)`
rot = np.asarray(res_jl.rot)

# Smoke test assertions
assert pos.shape == (3,), f"Expected position shape (3,), got {pos.shape}"
assert rot.shape == (3, 3), f"Expected rotation shape (3, 3), got {rot.shape}"
print(f"✓ Pose estimation test passed! Position: {pos}, Rotation shape: {rot.shape}")

# === Integrity Monitoring Test ===
# Use the estimated pose to compute integrity statistic
pf = jl.PointFeatures(points3d, points2d, camconf)
lf = jl.LineFeatures(line_pts, observed_lines, camconf)

# Test with points only
integrity_result = jl.compute_integrity_statistic(res_jl.pos, res_jl.rot, pf)
assert hasattr(integrity_result, 'stat'), "Expected 'stat' field in integrity result"
assert hasattr(integrity_result, 'p_value'), "Expected 'p_value' field in integrity result"
assert hasattr(integrity_result, 'dofs'), "Expected 'dofs' field in integrity result"
print(f"✓ Integrity (points only) - stat: {integrity_result.stat:.4f}, p-value: {integrity_result.p_value:.4f}, dofs: {integrity_result.dofs}")

# Test with points and lines
integrity_result_combined = jl.compute_integrity_statistic(res_jl.pos, res_jl.rot, pf, lf)
assert integrity_result_combined.dofs > integrity_result.dofs, "Expected more DOFs with lines"
print(f"✓ Integrity (points+lines) - stat: {integrity_result_combined.stat:.4f}, p-value: {integrity_result_combined.p_value:.4f}, dofs: {integrity_result_combined.dofs}")

print("✓ All smoke tests passed!")
