using Test
using RunwayLib
import RunwayLib: _ustrip, validate_camera_matrix, getopticalcenter
using Unitful, Unitful.DefaultSymbols
using StaticArrays
using Rotations

@testset "Camera Model" begin
    @testset "Projection Functions" begin
        @testset "Perfectly centered" begin
            # Test basic projection geometry with units
            cam_pos = WorldPoint(-1000.0u"m", 0.0u"m", 0.0u"m")
            cam_rot = RotZYX(0.0, 0.0, 0.0)
            world_pt = WorldPoint(0.0u"m", 0.0u"m", 0.0u"m")
            #
            # Test with both coordinate systems
            proj_pt_centered = project(cam_pos, cam_rot, world_pt, CAMERA_CONFIG_CENTERED)
            proj_pt_offset = project(cam_pos, cam_rot, world_pt, CAMERA_CONFIG_OFFSET)
            #
            @test proj_pt_centered ≈ ProjectionPoint(0px, 0px)
            @test proj_pt_offset ≈ let conf = CAMERA_CONFIG_OFFSET
                oc = getopticalcenter(conf)
                ProjectionPoint(oc[1], oc[2])
            end
        end

        @testset "Check signs: rotated" begin
            # Test basic projection geometry with units
            cam_pos = WorldPoint(-1000.0m, 0.0m, 0.0m)
            # slightly tilted nose up and right
            cam_rot = RotZYX(roll = 0°, pitch = 0°, yaw = 0°)
            world_pt = WorldPoint(0.0m, 10.0m, 10.0m)
            #
            # Test with both coordinate systems
            proj_pt_centered = project(cam_pos, cam_rot, world_pt, CAMERA_CONFIG_CENTERED)
            proj_pt_offset = project(cam_pos, cam_rot, world_pt, CAMERA_CONFIG_OFFSET)
            #
            @test proj_pt_centered.x > 0px
            @test proj_pt_centered.y > 0px
            #
            oc = getopticalcenter(CAMERA_CONFIG_OFFSET)
            @test proj_pt_offset.x < oc[1]
            @test proj_pt_offset.y < oc[2]
        end

        @testset "Check signs: roll" begin
            # Test basic projection geometry with units
            cam_pos = WorldPoint(-1000.0m, 0.0m, 0.0m)
            # slightly tilted nose up and right
            cam_rot = RotZYX(roll = 90°, pitch = 0°, yaw = 0°)
            world_pt = WorldPoint(0.0m, 10.0m, 10.0m)
            #
            # Test with both coordinate systems
            proj_pt_centered = project(cam_pos, cam_rot, world_pt, CAMERA_CONFIG_CENTERED)
            proj_pt_offset = project(cam_pos, cam_rot, world_pt, CAMERA_CONFIG_OFFSET)
            #
            @test proj_pt_centered.x > 0px
            @test proj_pt_centered.y < 0px
            #
            oc = getopticalcenter(CAMERA_CONFIG_OFFSET)
            @test proj_pt_offset.x < oc[1]
            @test proj_pt_offset.y > oc[2]
        end

        @testset "Check signs: pitch and yaw" begin
            # Test basic projection geometry with units
            cam_pos = WorldPoint(-1000.0m, 0.0m, 0.0m)
            # slightly tilted nose up and right
            cam_rot = RotZYX(roll = 0°, pitch = -1°, yaw = -1°)
            world_pt = WorldPoint(0.0m, 0.0m, 0.0m)
            #
            # Test with both coordinate systems
            proj_pt_centered = project(cam_pos, cam_rot, world_pt, CAMERA_CONFIG_CENTERED)
            proj_pt_offset = project(cam_pos, cam_rot, world_pt, CAMERA_CONFIG_OFFSET)
            #
            @test proj_pt_centered.x > 0px
            @test proj_pt_centered.y < 0px
            #
            oc = getopticalcenter(CAMERA_CONFIG_OFFSET)
            @test proj_pt_offset.x < oc[1]
            @test proj_pt_offset.y > oc[2]
        end

        @testset "Check throws behind camera" begin
            # Test basic projection geometry with units
            cam_pos = WorldPoint(0.0m, 0.0m, 0.0m)
            # slightly tilted nose up and right
            cam_rot = RotZYX(roll = 0°, pitch = -1°, yaw = -1°)
            world_pt = WorldPoint(-1.0m, 0.0m, 0.0m)
            @test_throws BehindCameraException project(cam_pos, cam_rot, world_pt, CAMERA_CONFIG_CENTERED)
            @test_throws BehindCameraException project(cam_pos, cam_rot, world_pt, CAMERA_CONFIG_OFFSET)
        end
    end

    # @testset "Camera Calibration" begin
    #     # Test focal length conversions with units
    #     focal_length = 25.0u"mm"
    #     pixel_size = 3.45u"μm" / pixel

    #     focal_length_px = focal_length / pixel_size
    #     @test uconvert(pixel, focal_length_px) ≈ 7246.377 * pixel rtol = 1.0e-3

    #     # Test field of view calculations
    #     image_width_px = 4096 * pixel
    #     sensor_width = image_width_px * pixel_size
    #     @test uconvert(u"mm", sensor_width) ≈ 14.1312u"mm" rtol = 1.0e-3

    #     # Calculate horizontal field of view
    #     sensor_width_m = uconvert(u"m", sensor_width)
    #     focal_length_m = uconvert(u"m", focal_length)
    #     horizontal_fov_rad = 2 * atan(ustrip(sensor_width_m) / (2 * ustrip(focal_length_m)))
    #     horizontal_fov_deg = rad2deg(horizontal_fov_rad)

    #     @test horizontal_fov_deg > 0
    #     @test horizontal_fov_deg < 180  # Should be reasonable FOV
    #     @test horizontal_fov_deg ≈ 31.0 rtol = 0.1  # Approximately 31 degrees for this setup

    #     # Test typical calibration parameter ranges with units using StaticArrays
    #     focal_lengths = SA[16.0u"mm", 25.0u"mm", 35.0u"mm", 50.0u"mm", 85.0u"mm"]
    #     pixel_sizes = SA[1.4u"μm", 2.2u"μm", 3.45u"μm", 5.5u"μm", 9.0u"μm"]

    #     for f in focal_lengths
    #         @test f > 0.0u"mm"
    #         @test f < 200.0u"mm"  # Reasonable range for aircraft cameras
    #     end

    #     for p in pixel_sizes
    #         @test p > 0.0u"μm"
    #         @test p < 20.0u"μm"  # Reasonable range for modern sensors
    #     end
    # end

    # @testset "Inverse Projection" begin
    #     # Test inverse projection concepts with units
    #     cam_pos = WorldPoint(-1000.0u"m", 0.0u"m", 100.0u"m")
    #     cam_rot = RotZYX(0.0, 0.0, 0.0)

    #     # Project a point
    #     world_pt = WorldPoint(0.0u"m", 10.0u"m", 0.0u"m")
    #     proj_pt = project(cam_pos, cam_rot, world_pt)

    #     # The projection should be deterministic
    #     proj_pt2 = project(cam_pos, cam_rot, world_pt)
    #     @test proj_pt.x ≈ proj_pt2.x
    #     @test proj_pt.y ≈ proj_pt2.y

    #     # Test pixel to ray direction concepts
    #     pixel_point_offset = ProjectionPoint{typeof(1.0pixel), :offset}(2048.0 * pixel, 1500.0 * pixel)
    #     pixel_point_centered = ProjectionPoint{typeof(1.0pixel), :centered}(0.0 * pixel, 0.0 * pixel)

    #     ray_dir_offset = pixel_to_ray_direction(pixel_point_offset, CAMERA_CONFIG_OFFSET)
    #     ray_dir_centered = pixel_to_ray_direction(pixel_point_centered, CAMERA_CONFIG_CENTERED)

    #     @test isa(ray_dir_offset, CameraPoint)
    #     @test isa(ray_dir_centered, CameraPoint)

    #     focal_length = 25.0u"mm"
    #     pixel_size = 3.45u"μm" / pixel

    #     # Test f-number calculations
    #     aperture_diameter = 5.0u"mm"
    #     f_number = focal_length / aperture_diameter  # f/5 for 25mm lens
    #     @test f_number ≈ 5.0 rtol = 1.0e-3
    # end

    @testset "CameraMatrix Functionality" begin
        @testset "CameraMatrix Construction" begin
            # Test valid matrix construction with proper units
            # Use the CameraMatrix constructor to create a proper matrix
            matrix = CameraMatrix(CAMERA_CONFIG_OFFSET).matrix

            # Test both coordinate systems
            cam_offset = CameraMatrix{:offset}(matrix, 2048.0px, 1024.0px)
            cam_centered = CameraMatrix{:centered}(matrix, 2048.0px, 1024.0px)

            @test cam_offset.matrix == matrix
            @test cam_offset.image_width == 2048.0px
            @test cam_offset.image_height == 1024.0px

            @test cam_centered.matrix == matrix
            @test cam_centered.image_width == 2048.0px
            @test cam_centered.image_height == 1024.0px

            # Test convenience constructor
            cam_convenience = CameraMatrix(:offset, matrix, 2048.0px, 1024.0px)
            @test cam_convenience.matrix == matrix
            @test cam_convenience.image_width == 2048.0px
        end

        @testset "CameraMatrix Validation" begin
            # Test invalid coordinate system
            matrix = CameraMatrix(CAMERA_CONFIG_OFFSET).matrix
            @test_throws ArgumentError CameraMatrix{:invalid}(matrix, 2048.0px, 1024.0px)

            # Test zero focal length - create by copying good matrix and setting focal lengths to zero
            bad_matrix1 = @SMatrix [
                matrix[1, 2]  matrix[1, 2]  matrix[1, 3];
                matrix[1, 2]  matrix[2, 2]  matrix[2, 3];
                matrix[3, 1]  matrix[3, 2]  matrix[3, 3]
            ]
            @test_throws ArgumentError CameraMatrix{:offset}(bad_matrix1, 2048.0px, 1024.0px)

            # Test invalid bottom row - create by copying good matrix and changing bottom row
            bad_matrix2 = @SMatrix [
                matrix[1, 1]  matrix[1, 2]  matrix[1, 3];
                matrix[2, 1]  matrix[2, 2]  matrix[2, 3];
                matrix[1, 1]  matrix[3, 2]  matrix[3, 3]
            ]
            @test_throws ArgumentError CameraMatrix{:offset}(bad_matrix2, 2048.0px, 1024.0px)

            # Test negative image dimensions
            @test_throws ArgumentError CameraMatrix{:offset}(matrix, -1.0px, 1024.0px)
            @test_throws ArgumentError CameraMatrix{:offset}(matrix, 2048.0px, 0.0px)
        end

        @testset "CameraMatrix Projection" begin
            # Create a camera matrix equivalent to CAMERA_CONFIG_OFFSET using the constructor
            cam_matrix = CameraMatrix(CAMERA_CONFIG_OFFSET)

            # Test projection with both camera types
            cam_pos = WorldPoint(-1000.0m, 0.0m, 0.0m)
            cam_rot = RotZYX(0.0, 0.0, 0.0)
            world_pt = WorldPoint(0.0m, 0.0m, 0.0m)

            proj_config = project(cam_pos, cam_rot, world_pt, CAMERA_CONFIG_OFFSET)
            proj_matrix = project(cam_pos, cam_rot, world_pt, cam_matrix)

            # Results should be very close (allowing for floating point precision)
            # Use ustrip to handle potential unit differences
            @test abs(ustrip(proj_config.x) - ustrip(proj_matrix.x)) < 1.0e-6
            @test abs(ustrip(proj_config.y) - ustrip(proj_matrix.y)) < 1.0e-6

            # Test with a point that's off-center
            world_pt2 = WorldPoint(0.0m, 10.0m, 10.0m)
            proj_config2 = project(cam_pos, cam_rot, world_pt2, CAMERA_CONFIG_OFFSET)
            proj_matrix2 = project(cam_pos, cam_rot, world_pt2, cam_matrix)

            @test abs(ustrip(proj_config2.x) - ustrip(proj_matrix2.x)) < 1.0e-6
            @test abs(ustrip(proj_config2.y) - ustrip(proj_matrix2.y)) < 1.0e-6
        end

        @testset "CameraMatrix(CameraConfig) Conversion" begin
            # Test conversion from CameraConfig to CameraMatrix
            matrix_config_offset = CameraMatrix(CAMERA_CONFIG_OFFSET)
            matrix_config_centered = CameraMatrix(CAMERA_CONFIG_CENTERED)

            # Test that conversions maintain coordinate system
            @test matrix_config_offset isa CameraMatrix{:offset}
            @test matrix_config_centered isa CameraMatrix{:centered}

            # Test that converted matrix produces same results
            cam_pos = WorldPoint(-1000.0m, 0.0m, 100.0m)
            cam_rot = RotZYX(0.0, 0.1, 0.0)
            world_pt = WorldPoint(100.0m, 25.0m, 0.0m)

            # Compare original vs converted for offset coordinates
            proj_orig_offset = project(cam_pos, cam_rot, world_pt, CAMERA_CONFIG_OFFSET)
            proj_conv_offset = project(cam_pos, cam_rot, world_pt, matrix_config_offset)

            @test abs(ustrip(proj_orig_offset.x) - ustrip(proj_conv_offset.x)) < 1.0e-6
            @test abs(ustrip(proj_orig_offset.y) - ustrip(proj_conv_offset.y)) < 1.0e-6

            # Compare original vs converted for centered coordinates
            proj_orig_centered = project(cam_pos, cam_rot, world_pt, CAMERA_CONFIG_CENTERED)
            proj_conv_centered = project(cam_pos, cam_rot, world_pt, matrix_config_centered)

            @test abs(ustrip(proj_orig_centered.x) - ustrip(proj_conv_centered.x)) < 1.0e-6
            @test abs(ustrip(proj_orig_centered.y) - ustrip(proj_conv_centered.y)) < 1.0e-6
        end

        @testset "Camera Matrix Validation Function" begin
            # Test valid matrix - using Float64 3x3 matrix for validate_camera_matrix function
            good_matrix = SA[
                2000.0  0.0     1024.0;
                0.0     2000.0  512.0;
                0.0     0.0     1.0
            ]px
            @test validate_camera_matrix(good_matrix) == true

            # Test matrix with zero focal length
            bad_focal = SA[
                0.0     0.0     1024.0;
                0.0     2000.0  512.0;
                0.0     0.0     1.0
            ]px
            @test validate_camera_matrix(bad_focal) == false

            # Test matrix with invalid bottom row
            bad_bottom = SA[
                2000.0  0.0     1024.0;
                0.0     2000.0  512.0;
                1.0     0.0     1.0
            ]px
            @test validate_camera_matrix(bad_bottom) == false

            # Test matrix with NaN
            nan_matrix = SA[
                NaN     0.0     1024.0;
                0.0     2000.0  512.0;
                0.0     0.0     1.0
            ]px
            @test validate_camera_matrix(nan_matrix) == false
        end

        @testset "Pose Estimation with CameraMatrix" begin
            # Test that pose estimation works with CameraMatrix
            runway_corners = [
                WorldPoint(0.0m, -50.0m, 0.0m),
                WorldPoint(0.0m, 50.0m, 0.0m),
                WorldPoint(1500.0m, -50.0m, 0.0m),
                WorldPoint(1500.0m, 50.0m, 0.0m),
            ]

            # Create projections using the original camera config
            true_pos = WorldPoint(-1300.0m, 0.0m, 80.0m)
            true_rot = RotZYX(yaw = 0.05, pitch = 0.04, roll = 0.03)

            projections = [project(true_pos, true_rot, corner, CAMERA_CONFIG_OFFSET) for corner in runway_corners]

            # Convert camera config to matrix
            camera_matrix = CameraMatrix(CAMERA_CONFIG_OFFSET)

            # Test 6DOF estimation with matrix
            pose_matrix = estimatepose6dof(runway_corners, projections, camera_matrix)
            pose_config = estimatepose6dof(runway_corners, projections, CAMERA_CONFIG_OFFSET)

            # Results should be very similar
            @test abs(pose_matrix.pos.x - pose_config.pos.x) < 1.0m
            @test abs(pose_matrix.pos.y - pose_config.pos.y) < 1.0m
            @test abs(pose_matrix.pos.z - pose_config.pos.z) < 1.0m

            # Test 3DOF estimation with matrix
            pose3d_matrix = estimatepose3dof(runway_corners, projections, true_rot, camera_matrix)
            pose3d_config = estimatepose3dof(runway_corners, projections, true_rot, CAMERA_CONFIG_OFFSET)

            # Results should be very similar
            @test abs(pose3d_matrix.pos.x - pose3d_config.pos.x) < 1.0m
            @test abs(pose3d_matrix.pos.y - pose3d_config.pos.y) < 1.0m
            @test abs(pose3d_matrix.pos.z - pose3d_config.pos.z) < 1.0m
        end
    end
end
