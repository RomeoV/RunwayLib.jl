using Test
using RunwayLib
using StaticArrays
using Rotations
using LinearAlgebra
using Unitful

@testset "Coordinate Systems" begin
    @testset "Coordinate Transformations" begin
        # Test world to camera transformation with units
        cam_pos = WorldPoint(0.0u"m", 0.0u"m", 0.0u"m")
        cam_rot = RotZYX(0.0, 0.0, 0.0)  # Identity rotation
        world_pt = WorldPoint(1.0u"m", 2.0u"m", 3.0u"m")

        # With identity transformation, camera point should equal world point
        cam_pt = world_pt_to_cam_pt(cam_pos, cam_rot, world_pt)
        @test cam_pt ≈ CameraPoint(1.0u"m", 2.0u"m", 3.0u"m")

        # Test inverse transformation
        world_pt_back = cam_pt_to_world_pt(cam_pos, cam_rot, cam_pt)
        @test world_pt_back ≈ world_pt

        # Test with non-trivial transformation
        cam_pos = WorldPoint(10.0u"m", 20.0u"m", 30.0u"m")
        cam_rot = RotZYX(roll = 0.0, pitch = 0.0, yaw = π / 2)  # 90 degree yaw rotation
        world_pt = WorldPoint(11.0u"m", 20.0u"m", 30.0u"m")  # 1 unit in x from camera

        cam_pt = world_pt_to_cam_pt(cam_pos, cam_rot, world_pt)
        # After 90 degree yaw rotation, x becomes -y in camera frame
        @test cam_pt.y ≈ -1.0u"m" atol = 1.0e-10u"m"
        @test abs(cam_pt.x) < 1.0e-10u"m"
        @test abs(cam_pt.z) < 1.0e-10u"m"

        # Test typical camera parameters with units
        focal_length = 25.0u"mm"
        pixel_size = 3.45u"μm" / pixel
        @test focal_length > 0.0u"mm"
        @test pixel_size > 0.0u"μm" / pixel
        @test uconvert(u"m", focal_length) ≈ 0.025u"m"
        @test uconvert(u"m" / pixel, pixel_size) ≈ 3.45e-6u"m" / pixel
    end

    @testset "Camera Projection" begin
        # Test basic projection with units
        cam_pos = WorldPoint(0.0u"m", 0.0u"m", 0.0u"m")
        cam_rot = RotZYX(0.0, 0.0, 0.0)
        world_pt = WorldPoint(1.0u"m", 0.1u"m", 0.1u"m")  # Point in front of camera

        # Test with default (offset) coordinates
        proj_pt = project(cam_pos, cam_rot, world_pt)

        # Test with :offset coordinate system
        proj_pt_offset = project(cam_pos, cam_rot, world_pt, CAMERA_CONFIG_OFFSET)

        # Test that points at camera position cannot be projected (division by zero)
        world_pt_at_camera = WorldPoint(0.0u"m", 0.0u"m", 0.0u"m")
        @test_throws BehindCameraException project(cam_pos, cam_rot, world_pt_at_camera, CAMERA_CONFIG_OFFSET)

        # Test projection consistency - points further away should project closer to center
        world_pt_near = WorldPoint(1.0u"m", 0.1u"m", 0.0u"m")
        world_pt_far = WorldPoint(10.0u"m", 1.0u"m", 0.0u"m")  # Same angle but further

        proj_near = project(cam_pos, cam_rot, world_pt_near, CAMERA_CONFIG_OFFSET)
        proj_far = project(cam_pos, cam_rot, world_pt_far, CAMERA_CONFIG_OFFSET)

        # Both should have same projection (similar triangles)
        @test proj_near.x ≈ proj_far.x atol = 1.0e-6 * 1pixel
        @test proj_near.y ≈ proj_far.y atol = 1.0e-6 * 1pixel

        # Test coordinate conversion (identity for same coordinate system)
        p = ProjectionPoint(:offset, rand(2)px...)
        p′ = convertcamconf(CAMERA_CONFIG_OFFSET, CAMERA_CONFIG_OFFSET, p)
        @test p ≈ p′  # Should be identity since both are :offset
    end
end
