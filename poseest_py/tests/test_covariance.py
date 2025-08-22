"""
Tests for covariance specification functionality in Python interface.

This module tests:
1. Covariance specification classes
2. Validation of covariance data
3. Integration with C API functions
4. End-to-end covariance specification workflow
"""

import unittest
import numpy as np
from poseest import (
    WorldPoint, ProjectionPoint, Rotation, CameraMatrix,
    DefaultCovariance, ScalarCovariance, DiagonalCovariance, BlockDiagonalCovariance, FullCovariance,
    estimate_pose_6dof, estimate_pose_3dof, project_point,
    InvalidInputError, InsufficientPointsError
)


class TestCovarianceSpecification(unittest.TestCase):
    """Test covariance specification classes and validation."""
    
    def setUp(self):
        """Set up test data."""
        self.runway_corners = [
            WorldPoint(1000.0, -50.0, 0.0),
            WorldPoint(1000.0, 50.0, 0.0),
            WorldPoint(3000.0, 50.0, 0.0),
            WorldPoint(3000.0, -50.0, 0.0),
        ]
        
        self.projections = [
            ProjectionPoint(100.0, 200.0),
            ProjectionPoint(150.0, 200.0),
            ProjectionPoint(150.0, 250.0),
            ProjectionPoint(100.0, 250.0),
        ]
        
        self.known_rotation = Rotation(yaw=0.05, pitch=0.03, roll=0.01)
    
    def test_default_covariance(self):
        """Test default covariance specification."""
        # Valid default covariance
        cov = DefaultCovariance()
        cov.validate(num_points=4)
        
        data, cov_type = cov.to_c_array(num_points=4)
        self.assertEqual(len(data), 1)  # Dummy data
        self.assertEqual(cov_type.value, 0)  # COV_DEFAULT
    
    def test_scalar_covariance(self):
        """Test scalar covariance specification."""
        # Valid scalar covariance
        cov = ScalarCovariance(noise_std=2.5)
        cov.validate(num_points=4)
        
        data, cov_type = cov.to_c_array(num_points=4)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0], 2.5)
        self.assertEqual(cov_type.value, 1)  # COV_SCALAR
        
        # Invalid scalar covariance
        with self.assertRaises(ValueError):
            invalid_cov = ScalarCovariance(noise_std=-1.0)
            invalid_cov.validate(num_points=4)
    
    def test_diagonal_covariance(self):
        """Test diagonal covariance specification."""
        # Valid diagonal covariance
        variances = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]  # 4 points * 2 coords
        cov = DiagonalCovariance(variances=variances)
        cov.validate(num_points=4)
        
        data, cov_type = cov.to_c_array(num_points=4)
        self.assertEqual(len(data), 8)
        self.assertEqual(list(data), variances)
        self.assertEqual(cov_type.value, 2)  # COV_DIAGONAL_FULL
        
        # Wrong number of variances
        with self.assertRaises(ValueError):
            invalid_cov = DiagonalCovariance(variances=[1.0, 2.0, 3.0])  # Too few
            invalid_cov.validate(num_points=4)
        
        # Negative variance
        with self.assertRaises(ValueError):
            invalid_cov = DiagonalCovariance(variances=[1.0, -1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
            invalid_cov.validate(num_points=4)
    
    def test_block_diagonal_covariance(self):
        """Test block diagonal covariance specification."""
        # Valid block diagonal covariance
        block_matrices = [
            [[1.0, 0.1], [0.1, 1.0]],    # Point 1
            [[2.0, 0.2], [0.2, 2.0]],    # Point 2
            [[1.5, 0.0], [0.0, 1.5]],    # Point 3
            [[2.5, -0.3], [-0.3, 2.5]]   # Point 4
        ]
        cov = BlockDiagonalCovariance(block_matrices=block_matrices)
        cov.validate(num_points=4)
        
        data, cov_type = cov.to_c_array(num_points=4)
        self.assertEqual(len(data), 16)  # 4 points * 4 elements per 2x2 matrix
        
        # Check flattening (row-major order)
        expected = [1.0, 0.1, 0.1, 1.0, 2.0, 0.2, 0.2, 2.0,
                   1.5, 0.0, 0.0, 1.5, 2.5, -0.3, -0.3, 2.5]
        self.assertEqual(list(data), expected)
        self.assertEqual(cov_type.value, 3)  # COV_BLOCK_DIAGONAL
        
        # Wrong number of matrices
        with self.assertRaises(ValueError):
            invalid_cov = BlockDiagonalCovariance(block_matrices=[[[1.0, 0.0], [0.0, 1.0]]])
            invalid_cov.validate(num_points=4)
        
        # Wrong matrix size
        with self.assertRaises(ValueError):
            invalid_matrices = [
                [[1.0, 0.1, 0.2], [0.1, 1.0, 0.3], [0.2, 0.3, 1.0]],  # 3x3 instead of 2x2
                [[2.0, 0.0], [0.0, 2.0]],
                [[1.5, 0.0], [0.0, 1.5]],
                [[2.5, 0.0], [0.0, 2.5]]
            ]
            invalid_cov = BlockDiagonalCovariance(block_matrices=invalid_matrices)
            invalid_cov.validate(num_points=4)
        
        # Non-positive-definite matrix
        with self.assertRaises(ValueError):
            invalid_matrices = [
                [[1.0, 2.0], [2.0, 1.0]],  # Determinant = -3 < 0
                [[2.0, 0.0], [0.0, 2.0]],
                [[1.5, 0.0], [0.0, 1.5]],
                [[2.5, 0.0], [0.0, 2.5]]
            ]
            invalid_cov = BlockDiagonalCovariance(block_matrices=invalid_matrices)
            invalid_cov.validate(num_points=4)
    
    def test_full_covariance(self):
        """Test full covariance matrix specification."""
        # Valid full covariance (8x8 identity matrix with some correlations)
        matrix_size = 8
        matrix = np.eye(matrix_size).tolist()
        matrix[0][1] = matrix[1][0] = 0.1  # Add correlation
        matrix[2][3] = matrix[3][2] = 0.2  # Add correlation
        
        cov = FullCovariance(matrix=matrix)
        cov.validate(num_points=4)
        
        data, cov_type = cov.to_c_array(num_points=4)
        self.assertEqual(len(data), 64)  # 8x8 matrix
        self.assertEqual(cov_type.value, 4)  # COV_FULL_MATRIX
        
        # Check flattening (row-major order)
        expected_flat = []
        for row in matrix:
            expected_flat.extend(row)
        self.assertEqual(list(data), expected_flat)
        
        # Wrong matrix size
        with self.assertRaises(ValueError):
            wrong_size_matrix = np.eye(6).tolist()  # 6x6 instead of 8x8
            invalid_cov = FullCovariance(matrix=wrong_size_matrix)
            invalid_cov.validate(num_points=4)
        
        # Non-square matrix
        with self.assertRaises(ValueError):
            non_square = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]  # 3x2 matrix
            invalid_cov = FullCovariance(matrix=non_square)
            invalid_cov.validate(num_points=4)
        
        # Non-symmetric matrix
        with self.assertRaises(ValueError):
            non_symmetric = np.eye(8).tolist()
            non_symmetric[0][1] = 0.1
            non_symmetric[1][0] = 0.2  # Different from [0][1]
            invalid_cov = FullCovariance(matrix=non_symmetric)
            invalid_cov.validate(num_points=4)
        
        # Non-positive-definite matrix (negative diagonal element)
        with self.assertRaises(ValueError):
            non_pos_def = np.eye(8).tolist()
            non_pos_def[0][0] = -1.0  # Negative diagonal element
            invalid_cov = FullCovariance(matrix=non_pos_def)
            invalid_cov.validate(num_points=4)


class TestCovarianceIntegration(unittest.TestCase):
    """Test integration of covariance specifications with pose estimation."""
    
    def setUp(self):
        """Set up test data with realistic aircraft pose and runway geometry."""
        # Standard runway corners (4 points forming a rectangle) - matches Julia tests
        self.runway_corners = [
            WorldPoint(0.0, 25.0, 0.0),      # near left  
            WorldPoint(0.0, -25.0, 0.0),     # near right
            WorldPoint(1000.0, 25.0, 0.0),   # far left
            WorldPoint(1000.0, -25.0, 0.0),  # far right
        ]
        
        # Realistic aircraft pose - behind runway, moderate altitude
        self.true_position = WorldPoint(-500.0, 10.0, 100.0)
        self.known_rotation = Rotation(yaw=-0.01, pitch=0.1, roll=0.02)
        
        # Create equivalent camera matrix for offset coordinate system
        # Based on CAMERA_CONFIG_OFFSET: 25mm focal length, 3.45μm pixel size, 4096x3000 image
        focal_length_px = 25.0 / (3.45e-3)  # ~7246 pixels
        cx = (4096 + 1) / 2  # 2048.5
        cy = (3000 + 1) / 2  # 1500.5
        
        self.camera_matrix = CameraMatrix(
            matrix=[
                [-focal_length_px, 0.0, cx],  # negative for offset coordinate system
                [0.0, -focal_length_px, cy],
                [0.0, 0.0, 1.0]
            ],
            image_width=4096.0,
            image_height=3000.0,
            coordinate_system='offset'
        )
        
        # Generate realistic projections using the project_point function
        self.projections = []
        for corner in self.runway_corners:
            proj = project_point(
                camera_position=self.true_position,
                camera_rotation=self.known_rotation, 
                world_point=corner,
                camera_matrix=self.camera_matrix
            )
            self.projections.append(proj)
    
    def test_6dof_with_default_covariance(self):
        """Test 6DOF estimation with explicit default covariance."""
        cov = DefaultCovariance()
        
        result = estimate_pose_6dof(
            runway_corners=self.runway_corners,
            projections=self.projections,
            camera_matrix=self.camera_matrix,
            covariance=cov
        )
        
        # Check that result is close to true pose (within reasonable tolerance)
        pos_error = ((result.position.x - self.true_position.x)**2 + 
                     (result.position.y - self.true_position.y)**2 + 
                     (result.position.z - self.true_position.z)**2)**0.5
        self.assertLess(pos_error, 1.0, "Position error should be less than 1 meter")
        
        # Check rotation is reasonably close (within 0.01 radians ~ 0.6 degrees)
        rot_error = abs(result.rotation.yaw - self.known_rotation.yaw) + \
                   abs(result.rotation.pitch - self.known_rotation.pitch) + \
                   abs(result.rotation.roll - self.known_rotation.roll)
        self.assertLess(rot_error, 0.01, "Rotation error should be less than 0.01 radians")
    
    def test_6dof_with_scalar_covariance(self):
        """Test 6DOF estimation with scalar covariance."""
        cov = ScalarCovariance(noise_std=2.0)
        
        result = estimate_pose_6dof(
            runway_corners=self.runway_corners,
            projections=self.projections,
            camera_matrix=self.camera_matrix,
            covariance=cov
        )
        
        # Check that result is close to true pose
        pos_error = ((result.position.x - self.true_position.x)**2 + 
                     (result.position.y - self.true_position.y)**2 + 
                     (result.position.z - self.true_position.z)**2)**0.5
        self.assertLess(pos_error, 1.0, "Position error should be less than 1 meter")
        
        # Check basic result properties
        self.assertIsInstance(result.residual, float)
        self.assertIsInstance(result.converged, bool)
    
    def test_6dof_with_diagonal_covariance(self):
        """Test 6DOF estimation with diagonal covariance."""
        # Different noise levels for each measurement
        variances = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
        cov = DiagonalCovariance(variances=variances)
        
        result = estimate_pose_6dof(
            runway_corners=self.runway_corners,
            projections=self.projections,
            camera_matrix=self.camera_matrix,
            covariance=cov
        )
        
        # Check that result is close to true pose
        pos_error = ((result.position.x - self.true_position.x)**2 + 
                     (result.position.y - self.true_position.y)**2 + 
                     (result.position.z - self.true_position.z)**2)**0.5
        self.assertLess(pos_error, 1.0, "Position error should be less than 1 meter")
    
    def test_3dof_with_block_diagonal_covariance(self):
        """Test 3DOF estimation with block diagonal covariance."""
        block_matrices = [
            [[2.0, 0.1], [0.1, 2.0]],    # Point 1 - some correlation
            [[1.5, 0.0], [0.0, 1.5]],    # Point 2 - no correlation
            [[3.0, -0.2], [-0.2, 3.0]],  # Point 3 - negative correlation
            [[2.5, 0.15], [0.15, 2.5]]   # Point 4 - positive correlation
        ]
        cov = BlockDiagonalCovariance(block_matrices=block_matrices)
        
        result = estimate_pose_3dof(
            runway_corners=self.runway_corners,
            projections=self.projections,
            known_rotation=self.known_rotation,
            camera_matrix=self.camera_matrix,
            covariance=cov
        )
        
        # Check that position is close to true position
        pos_error = ((result.position.x - self.true_position.x)**2 + 
                     (result.position.y - self.true_position.y)**2 + 
                     (result.position.z - self.true_position.z)**2)**0.5
        self.assertLess(pos_error, 1.0, "Position error should be less than 1 meter")
        
        # For 3DOF, rotation should be exactly the known rotation
        self.assertEqual(result.rotation.yaw, self.known_rotation.yaw)
        self.assertEqual(result.rotation.pitch, self.known_rotation.pitch)
        self.assertEqual(result.rotation.roll, self.known_rotation.roll)
    
    def test_backward_compatibility(self):
        """Test that functions work without covariance (backward compatibility)."""
        # Test 6DOF without covariance - should use defaults
        result_6dof = estimate_pose_6dof(
            runway_corners=self.runway_corners,
            projections=self.projections,
            camera_matrix=self.camera_matrix
        )
        
        # Check that 6DOF result is close to true pose
        pos_error = ((result_6dof.position.x - self.true_position.x)**2 + 
                     (result_6dof.position.y - self.true_position.y)**2 + 
                     (result_6dof.position.z - self.true_position.z)**2)**0.5
        self.assertLess(pos_error, 1.0, "6DOF position error should be less than 1 meter")
        
        # Test 3DOF without covariance - should use defaults
        result_3dof = estimate_pose_3dof(
            runway_corners=self.runway_corners,
            projections=self.projections,
            known_rotation=self.known_rotation,
            camera_matrix=self.camera_matrix
        )
        
        # Check that 3DOF position is close to true position
        pos_error_3dof = ((result_3dof.position.x - self.true_position.x)**2 + 
                          (result_3dof.position.y - self.true_position.y)**2 + 
                          (result_3dof.position.z - self.true_position.z)**2)**0.5
        self.assertLess(pos_error_3dof, 1.0, "3DOF position error should be less than 1 meter")
    
    def test_covariance_validation_integration(self):
        """Test that covariance validation is enforced during estimation."""
        # Test with invalid covariance - should fail before C API call
        invalid_cov = ScalarCovariance(noise_std=-1.0)
        
        with self.assertRaises(ValueError):
            estimate_pose_6dof(
                runway_corners=self.runway_corners,
                projections=self.projections,
                camera_matrix=self.camera_matrix,
                covariance=invalid_cov
            )
    
    def test_insufficient_points_with_covariance(self):
        """Test error handling with insufficient points and covariance."""
        cov = ScalarCovariance(noise_std=2.0)
        
        # Test 6DOF with insufficient points
        with self.assertRaises(InsufficientPointsError):
            estimate_pose_6dof(
                runway_corners=self.runway_corners[:3],  # Only 3 points
                projections=self.projections[:3],
                camera_matrix=self.camera_matrix,
                covariance=cov
            )
        
        # Test 3DOF with insufficient points
        with self.assertRaises(InsufficientPointsError):
            estimate_pose_3dof(
                runway_corners=self.runway_corners[:2],  # Only 2 points
                projections=self.projections[:2],
                known_rotation=self.known_rotation,
                camera_matrix=self.camera_matrix,
                covariance=cov
            )


class TestCovarianceConsistency(unittest.TestCase):
    """Test consistency between different covariance representations."""
    
    def setUp(self):
        """Set up test data."""
        self.runway_corners = [
            WorldPoint(1000.0, -50.0, 0.0),
            WorldPoint(1000.0, 50.0, 0.0),
            WorldPoint(3000.0, 50.0, 0.0),
            WorldPoint(3000.0, -50.0, 0.0),
        ]
        
        self.projections = [
            ProjectionPoint(320.0, 240.0),
            ProjectionPoint(380.0, 240.0),
            ProjectionPoint(380.0, 280.0),
            ProjectionPoint(320.0, 280.0),
        ]
    
    def test_scalar_vs_diagonal_consistency(self):
        """Test that scalar and diagonal covariance give same results for uniform noise."""
        noise_std = 2.0
        
        # Scalar covariance
        scalar_cov = ScalarCovariance(noise_std=noise_std)
        
        # Equivalent diagonal covariance
        variances = [noise_std**2] * 8  # 4 points * 2 coords
        diagonal_cov = DiagonalCovariance(variances=variances)
        
        # Both should convert to same underlying representation
        scalar_data, scalar_type = scalar_cov.to_c_array(num_points=4)
        diagonal_data, diagonal_type = diagonal_cov.to_c_array(num_points=4)
        
        # Different types but should represent same noise model
        self.assertEqual(scalar_type.value, 1)  # COV_SCALAR
        self.assertEqual(diagonal_type.value, 2)  # COV_DIAGONAL_FULL
        
        # Scalar should have single value equal to std
        self.assertEqual(scalar_data[0], noise_std)
        
        # Diagonal should have variances equal to std^2
        expected_variances = [noise_std**2] * 8
        self.assertEqual(list(diagonal_data), expected_variances)
    
    def test_block_diagonal_vs_full_consistency(self):
        """Test consistency between block diagonal and equivalent full matrix."""
        # Create block diagonal covariance
        block_matrices = [
            [[1.0, 0.1], [0.1, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
            [[1.5, -0.1], [-0.1, 1.5]],
            [[2.5, 0.2], [0.2, 2.5]]
        ]
        block_cov = BlockDiagonalCovariance(block_matrices=block_matrices)
        
        # Create equivalent full matrix (block diagonal)
        full_matrix = np.zeros((8, 8))
        for i, block in enumerate(block_matrices):
            start_idx = i * 2
            end_idx = start_idx + 2
            full_matrix[start_idx:end_idx, start_idx:end_idx] = block
        
        full_cov = FullCovariance(matrix=full_matrix.tolist())
        
        # Both should be valid
        block_cov.validate(num_points=4)
        full_cov.validate(num_points=4)
        
        # Convert to C arrays
        block_data, block_type = block_cov.to_c_array(num_points=4)
        full_data, full_type = full_cov.to_c_array(num_points=4)
        
        self.assertEqual(block_type.value, 3)  # COV_BLOCK_DIAGONAL
        self.assertEqual(full_type.value, 4)   # COV_FULL_MATRIX
        
        # The underlying matrices should be mathematically equivalent
        # (though represented differently in the C arrays)
        self.assertEqual(len(block_data), 16)  # 4 blocks * 4 elements
        self.assertEqual(len(full_data), 64)   # 8x8 matrix


if __name__ == '__main__':
    unittest.main()
