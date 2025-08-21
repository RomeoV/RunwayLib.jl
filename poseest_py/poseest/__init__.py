"""
PoseEst: Python wrapper for runway pose estimation using Julia/C library.

This package provides a clean Python interface to high-performance pose estimation
algorithms implemented in Julia and compiled to a relocatable C library.
"""

__version__ = "0.1.0"

# Import main classes and functions
from .core import (
    # Data types
    WorldPoint,
    ProjectionPoint, 
    Rotation,
    PoseEstimate,
    
    # Functions
    estimate_pose_6dof,
    estimate_pose_3dof,
    project_point,
    
    # Configuration
    CameraConfig,
    CameraMatrix,
    
    # Covariance specifications
    CovarianceSpec,
    DefaultCovariance,
    ScalarCovariance,
    DiagonalCovariance,
    BlockDiagonalCovariance,
    FullCovariance,
    CovarianceType,
    
    # Exceptions
    PoseEstimationError,
    InvalidInputError,
    BehindCameraError,
    ConvergenceError,
    InsufficientPointsError,
)

# Make everything available at package level
__all__ = [
    # Version
    "__version__",
    
    # Data types
    "WorldPoint",
    "ProjectionPoint",
    "Rotation", 
    "PoseEstimate",
    
    # Functions
    "estimate_pose_6dof",
    "estimate_pose_3dof", 
    "project_point",
    
    # Configuration
    "CameraConfig",
    "CameraMatrix",
    
    # Covariance specifications
    "CovarianceSpec",
    "DefaultCovariance",
    "ScalarCovariance",
    "DiagonalCovariance",
    "BlockDiagonalCovariance",
    "FullCovariance",
    "CovarianceType",
    
    # Exceptions
    "PoseEstimationError",
    "InvalidInputError", 
    "BehindCameraError",
    "ConvergenceError",
    "InsufficientPointsError",
]