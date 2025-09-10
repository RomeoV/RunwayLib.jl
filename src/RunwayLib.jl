module RunwayLib

using Distributions: Distributions, Normal, Chisq, ccdf
using LinearAlgebra: LinearAlgebra, /, cholesky
using LinearSolve: CholeskyFactorization, LinearSolve, NonlinearFunction,
    NonlinearLeastSquaresProblem, init
using Rotations: Rotations, RotZYX, Rotation
using ADTypes: AutoForwardDiff
using DifferentiationInterface: DifferentiationInterface, jacobian
using NonlinearSolveFirstOrder: LevenbergMarquardt, NonlinearLeastSquaresProblem, NonlinearFunction,
    reinit!, solve!
import StaticArrays: similar_type
using StaticArrays: StaticArrays, FieldVector, SA, Size, SVector
using TypedTables: TypedTables, Table
using Unitful: Unitful, @u_str, @unit, NoUnits, Quantity, dimension, uconvert,
    ustrip, Length
using Unitful.DefaultSymbols: DefaultSymbols, A, S, T, m, ps, rad, s
using ProbabilisticParameterEstimators: UncorrGaussianNoiseModel, CorrGaussianNoiseModel,
    NoiseModel, covmatrix
using SciMLBase: successful_retcode, FullSpecialize
import SciMLBase
import Base: OncePerTask

_uconvert(u) = Base.Fix1(uconvert, u)
_ustrip(u) = Base.Fix1(ustrip, u)
_reduce(f::F) where F = Base.Fix1(reduce, f)

# Define custom pixel unit
@unit pixel "pixel" Pixel 1 false
const px = pixel

# Register the pixel unit with Unitful
Unitful.register(RunwayLib)

# Export coordinate system types
export WorldPoint, CameraPoint, ProjectionPoint

# Export transformation functions
export world_pt_to_cam_pt, cam_pt_to_world_pt, project

# Export data structures
export RunwaySpec, PoseEstimate

# Re-export noise models from ProbabilisticParameterEstimators
export UncorrGaussianNoiseModel, CorrGaussianNoiseModel, NoiseModel, covmatrix

export parse_covariance_data

# Export camera model functions
export get_focal_length_pixels, get_field_of_view, pixel_to_ray_direction

# Export runway database functions
export get_runway_corners, validate_runway_spec

# Export TypedTables for data management
export Table

# Export configuration
export CAMERA_CONFIG, CAMERA_CONFIG_OFFSET, CameraConfig, DEFAULT_OPTIMIZATION_CONFIG, convertcamconf, CameraMatrix

# Export WithDims for flexible typing
export WithDims, WithUnits

# Export custom units
export pixel, px

export BehindCameraException

# Include submodules
include("coordinate_systems/types.jl")
include("coordinate_systems/transformations.jl")
include("pose_estimation/types.jl")
include("camera_model/withdims.jl")
include("camera_model/projection.jl")
include("camera_model/errors.jl")
# include("data_management/runway_database.jl")
include("pose_estimation/optimization.jl")
include("pose_estimation/errors.jl")
include("integrity/integrity.jl")
include("entrypoints.jl")
include("c_api.jl")

# Export pose estimation entrypoints and types
export estimatepose6dof, estimatepose3dof, pose_optimization
export PoseOptimizationParams6DOF, PoseOptimizationParams3DOF

# Export integrity monitoring functions
export compute_integrity_statistic, check_integrity, compute_jacobian, compute_residual

function load_runway_database(filename)
    error("load_runway_database not yet implemented")
end

function load_flight_data(filename)
    error("load_flight_data not yet implemented - will return TypedTables.Table")
end

function extract_runway_corners(flight_data_row)
    error("extract_runway_corners not yet implemented - accepts TypedTables row")
end

function extract_uncertainties(flight_data_row)
    error("extract_uncertainties not yet implemented - accepts TypedTables row")
end

# Include precompile workloads
include("precompile_workloads.jl")

end # module RunwayLib
