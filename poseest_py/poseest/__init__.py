"""
PoseEst: Python wrapper for runway pose estimation using Julia/C library.

This package provides a clean Python interface to high-performance pose estimation
algorithms implemented in Julia and compiled to a relocatable C library.
"""


import os
import sys
from pathlib import Path

# === ENVIRONMENT SETUP (Must be first!) ===
def _ensure_library_environment():
    """Ensure LD_LIBRARY_PATH is set before any library loading occurs."""

    # Find the path to our bundled Julia libraries
    package_dir = Path(__file__).parent
    julia_lib_dir = package_dir / "native" / "RunwayLibCompiled" / "lib" / "julia"
    julia_lib_str = str(julia_lib_dir)

    # Check if LD_LIBRARY_PATH already includes our path
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")

    if julia_lib_str not in current_ld_path.split(os.pathsep):
        # We need to set the environment and re-execute
        print(f"Setting up library environment...")

        # Prepend our path to LD_LIBRARY_PATH
        if current_ld_path:
            new_ld_path = f"{julia_lib_str}{os.pathsep}{current_ld_path}"
        else:
            new_ld_path = julia_lib_str

        os.environ["LD_LIBRARY_PATH"] = new_ld_path

        # Re-execute the current Python process with the updated environment
        try:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as e:
            print(f"Warning: Failed to re-execute with proper environment: {e}")
            print(f"You may need to set LD_LIBRARY_PATH={julia_lib_str}")
            # Continue anyway - maybe it will work

# Call this immediately when the package is imported
_ensure_library_environment()



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