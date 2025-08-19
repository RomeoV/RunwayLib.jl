"""
Comprehensive tests for the poseest Python package.

Uses the same test data and methodology as the Julia tests for consistency.
"""

import pytest
import sys
import os
import math
import random
from pathlib import Path

# Add the parent directory to the path so we can import poseest
sys.path.insert(0, str(Path(__file__).parent.parent))

import poseest


class TestDataTypes:
    """Test basic data type construction and conversion."""
    
    def test_world_point(self):
        """Test WorldPoint construction and properties."""
        wp = poseest.WorldPoint(100.0, -25.0, 50.0)
        assert wp.x == 100.0
        assert wp.y == -25.0
        assert wp.z == 50.0
    
    def test_projection_point(self):
        """Test ProjectionPoint construction and properties."""
        pp = poseest.ProjectionPoint(320.5, 240.2)
        assert pp.x == 320.5
        assert pp.y == 240.2
    
    def test_rotation(self):
        """Test Rotation construction and properties."""
        rot = poseest.Rotation(0.1, 0.05, -0.02)
        assert rot.yaw == 0.1
        assert rot.pitch == 0.05
        assert rot.roll == -0.02

    def test_c_struct_conversion(self):
        """Test conversion to/from C structures."""
        # Test WorldPoint conversion
        wp = poseest.WorldPoint(-500.0, 10.0, 80.0)
        wp_c = wp.to_c_struct()
        wp_back = poseest.WorldPoint.from_c_struct(wp_c)
        
        assert wp_back.x == wp.x
        assert wp_back.y == wp.y
        assert wp_back.z == wp.z
        
        # Test ProjectionPoint conversion
        pp = poseest.ProjectionPoint(245.3, 180.7)
        pp_c = pp.to_c_struct()
        pp_back = poseest.ProjectionPoint.from_c_struct(pp_c)
        
        assert pp_back.x == pp.x
        assert pp_back.y == pp.y
        
        # Test Rotation conversion  
        rot = poseest.Rotation(0.05, 0.04, 0.03)
        rot_c = rot.to_c_struct()
        rot_back = poseest.Rotation.from_c_struct(rot_c)
        
        assert rot_back.yaw == rot.yaw
        assert rot_back.pitch == rot.pitch
        assert rot_back.roll == rot.roll


class TestCameraConfig:
    """Test CameraConfig enum values."""
    
    def test_enum_values(self):
        """Test CameraConfig enum values."""
        assert poseest.CameraConfig.CENTERED == 0
        assert poseest.CameraConfig.OFFSET == 1


class TestLibraryIntegration:
    """Test integration with the native library using realistic test data."""
    
    def generate_runway_keypoints(self, num_keypoints):
        """Generate keypoints in left/right pairs along the runway."""
        runway_length = 3000.0
        runway_width = 100.0  # -50 to +50
        
        # Number of pairs along the runway
        num_pairs = num_keypoints // 2
        
        keypoints = []
        for i in range(num_pairs):
            x_pos = (i / (num_pairs - 1)) * runway_length
            # Add left point (-50 y)
            keypoints.append(poseest.WorldPoint(x_pos, -50.0, 0.0))
            # Add right point (+50 y)
            keypoints.append(poseest.WorldPoint(x_pos, 50.0, 0.0))
        
        return keypoints
    
    @pytest.fixture(params=[4, 8, 16])
    def test_data(self, request):
        """Generate test data using the same approach as Julia tests."""
        num_keypoints = request.param
        runway_corners = self.generate_runway_keypoints(num_keypoints)
        
        # True camera pose (same as Julia tests)
        true_position = poseest.WorldPoint(-1300.0, 0.0, 80.0)
        true_rotation = poseest.Rotation(yaw=0.05, pitch=0.04, roll=0.03)
        camera_config = poseest.CameraConfig.OFFSET
        
        # Generate realistic projections using the project function
        projections = []
        for corner in runway_corners:
            projection = poseest.project_point(
                true_position,
                true_rotation, 
                corner,
                camera_config
            )
            # Add small amount of noise (similar to Julia tests)
            noise_x = random.gauss(0, 0.5)  # 0.5 pixel standard deviation
            noise_y = random.gauss(0, 0.5)
            projection = poseest.ProjectionPoint(
                projection.x + noise_x,
                projection.y + noise_y
            )
            projections.append(projection)
        
        return {
            'runway_corners': runway_corners,
            'projections': projections,
            'true_position': true_position,
            'true_rotation': true_rotation,
            'camera_config': camera_config
        }
    
    def test_6dof_pose_estimation(self, test_data):
        """Test 6DOF pose estimation with realistic data."""
        # Set random seed for reproducible noise
        random.seed(42)
        
        num_keypoints = len(test_data['runway_corners'])
        print(f"Testing 6DOF pose estimation with {num_keypoints} keypoints")
        
        # Test with custom initial guess to ensure the parameter works
        initial_pos = poseest.WorldPoint(-2000.0, 10.0, 150.0)
        initial_rot = poseest.Rotation(0.1, 0.08, 0.02)
        
        pose = poseest.estimate_pose_6dof(
            test_data['runway_corners'],
            test_data['projections'],
            test_data['camera_config'],
            initial_guess_pos=initial_pos,
            initial_guess_rot=initial_rot
        )
        
        # Basic type checks
        assert isinstance(pose, poseest.PoseEstimate)
        assert isinstance(pose.position, poseest.WorldPoint)
        assert isinstance(pose.rotation, poseest.Rotation)
        assert isinstance(pose.residual, float)
        assert isinstance(pose.converged, bool)

        # The estimation should be reasonably close to the true pose
        # (allowing for noise and numerical precision)
        true_pos = test_data['true_position']
        true_rot = test_data['true_rotation']
        
        # Tolerance may be tighter with more keypoints
        tolerance_pos = 50.0 if num_keypoints == 4 else 30.0
        tolerance_rot = 0.1 if num_keypoints == 4 else 0.05
        
        assert abs(pose.position.x - true_pos.x) < tolerance_pos
        assert abs(pose.position.y - true_pos.y) < tolerance_pos  
        assert abs(pose.position.z - true_pos.z) < tolerance_pos
        
        assert abs(pose.rotation.yaw - true_rot.yaw) < tolerance_rot
        assert abs(pose.rotation.pitch - true_rot.pitch) < tolerance_rot
        assert abs(pose.rotation.roll - true_rot.roll) < tolerance_rot
        
        assert pose.converged == True
    
    def test_3dof_pose_estimation(self, test_data):
        """Test 3DOF pose estimation with known rotation."""
        num_keypoints = len(test_data['runway_corners'])
        print(f"Testing 3DOF pose estimation with {num_keypoints} keypoints")
        
        # Test with custom initial guess to ensure the parameter works
        initial_pos = poseest.WorldPoint(-2000.0, 10.0, 150.0)
        
        pose = poseest.estimate_pose_3dof(
            test_data['runway_corners'],
            test_data['projections'],
            test_data['true_rotation'],  # Known rotation
            test_data['camera_config'],
            initial_guess_pos=initial_pos
        )
        
        # Basic type checks
        assert isinstance(pose, poseest.PoseEstimate)
        assert isinstance(pose.position, poseest.WorldPoint)
        assert isinstance(pose.rotation, poseest.Rotation)
        
        # Position should be close to true position
        # More keypoints should give better accuracy
        tolerance_pos = 20.0 if num_keypoints == 4 else 10.0
        true_pos = test_data['true_position']
        assert abs(pose.position.x - true_pos.x) < tolerance_pos
        assert abs(pose.position.y - true_pos.y) < tolerance_pos
        assert abs(pose.position.z - true_pos.z) < tolerance_pos
        
        # Rotation should match the known rotation (exactly)
        true_rot = test_data['true_rotation']
        assert abs(pose.rotation.yaw - true_rot.yaw) < 1e-10
        assert abs(pose.rotation.pitch - true_rot.pitch) < 1e-10
        assert abs(pose.rotation.roll - true_rot.roll) < 1e-10
    
    def test_point_projection(self, test_data):
        """Test point projection functionality."""
        num_keypoints = len(test_data['runway_corners'])
        print(f"Testing point projection with {num_keypoints} keypoints")
        
        # Test projecting one of the runway corners
        corner = test_data['runway_corners'][0]
        true_pos = test_data['true_position']
        true_rot = test_data['true_rotation']
        
        projection = poseest.project_point(
            true_pos,
            true_rot,
            corner,
            test_data['camera_config']
        )
        
        # Basic type checks
        assert isinstance(projection, poseest.ProjectionPoint)
        assert isinstance(projection.x, float)
        assert isinstance(projection.y, float)
        
        # Projection should be reasonable (within typical image bounds)
        assert 0.0 < projection.x < 4096.0  # Typical image width
        assert 0.0 < projection.y < 3000.0  # Typical image height
        
        # Should be close to the generated test projection (within noise)
        test_projection = test_data['projections'][0]
        assert abs(projection.x - test_projection.x) < 2.0  # Within a few pixels
        assert abs(projection.y - test_projection.y) < 2.0


class TestErrorHandling:
    """Test error handling for invalid inputs."""
    
    def test_insufficient_points_6dof(self):
        """Test that insufficient points raises the right error for 6DOF."""
        runway_corners = [
            poseest.WorldPoint(0.0, -50.0, 0.0),
            poseest.WorldPoint(0.0, 50.0, 0.0),  # Only 2 points
        ]
        
        projections = [
            poseest.ProjectionPoint(100.0, 200.0),
            poseest.ProjectionPoint(110.0, 200.0),
        ]
        
        with pytest.raises(poseest.InsufficientPointsError):
            poseest.estimate_pose_6dof(
                runway_corners, 
                projections, 
                poseest.CameraConfig.OFFSET
            )
    
    def test_insufficient_points_3dof(self):
        """Test that insufficient points raises the right error for 3DOF."""
        runway_corners = [
            poseest.WorldPoint(0.0, -50.0, 0.0),
            poseest.WorldPoint(0.0, 50.0, 0.0),  # Only 2 points
        ]
        
        projections = [
            poseest.ProjectionPoint(100.0, 200.0),
            poseest.ProjectionPoint(110.0, 200.0),
        ]
        
        known_rotation = poseest.Rotation(0.0, 0.0, 0.0)
        
        with pytest.raises(poseest.InsufficientPointsError):
            poseest.estimate_pose_3dof(
                runway_corners, 
                projections, 
                known_rotation,
                poseest.CameraConfig.OFFSET
            )
    
    def test_mismatched_arrays(self):
        """Test that mismatched array sizes raise the right error."""
        runway_corners = [
            poseest.WorldPoint(0.0, -50.0, 0.0),
            poseest.WorldPoint(0.0, 50.0, 0.0),
            poseest.WorldPoint(3000.0, 50.0, 0.0),
            poseest.WorldPoint(3000.0, -50.0, 0.0),
        ]
        
        projections = [
            poseest.ProjectionPoint(100.0, 200.0),  # Only 1 projection for 4 corners
        ]
        
        with pytest.raises(poseest.InvalidInputError):
            poseest.estimate_pose_6dof(
                runway_corners, 
                projections, 
                poseest.CameraConfig.OFFSET
            )


if __name__ == "__main__":
    # Run the tests if executed directly
    pytest.main([__file__, "-v"])
