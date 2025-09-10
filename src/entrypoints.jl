Base.@ccallable function test_estimators()::Cint
    runway_corners = [
        WorldPoint(0.0m, -50.0m, 0.0m),
        WorldPoint(0.0m, 50.0m, 0.0m),
        WorldPoint(1500.0m, -50.0m, 0.0m),
        WorldPoint(1500.0m, 50.0m, 0.0m),
        WorldPoint(3000.0m, 50.0m, 0.0m),
        WorldPoint(3000.0m, -50.0m, 0.0m),
    ]
    true_pos = WorldPoint(-2300.0m, 0.0m, 80.0m)
    true_rot = RotZYX(roll=0.03, pitch=0.04, yaw=0.05)
    camconfig = CameraMatrix(CAMERA_CONFIG_OFFSET)
    projections = [
        project(true_pos, true_rot, corner, camconfig) + ProjectionPoint(randn(2)) * px
        for corner in runway_corners
    ]

    sol6dof = estimatepose6dof(runway_corners, projections, camconfig)

    # sol3dof = estimatepose3dof(
    #     runway_corners, projections, true_rot, camconfig
    # )
    println(Core.stdout, ustrip(m, sol6dof.pos.x))
    println(Core.stdout, ustrip(m, sol6dof.pos.y))
    println(Core.stdout, ustrip(m, sol6dof.pos.z))
    return 0
end
