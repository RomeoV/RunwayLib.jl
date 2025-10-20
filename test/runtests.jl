using Test
using RunwayLib
using StaticArrays

const GROUP = get(ENV, "GROUP", "All")

@testset "RunwayPoseEstimation.jl" begin
    # Julia unit tests (including tests for @ccallable Julia functions)
    if GROUP == "All" || GROUP == "Julia"
        include("unit/test_coordinate_systems.jl")
        include("unit/test_camera_model.jl")
        # include("unit/test_data_management.jl")
        include("unit/test_pose_estimation.jl")
        include("unit/test_c_api.jl")  # Tests @ccallable Julia functions from Julia
        include("unit/test_jet.jl")
        include("unit/test_covariance_specification.jl")
        # include("unit/test_uncertainty_quantification.jl")
        include("unit/test_integrity_monitoring.jl")
        # include("unit/test_visualization.jl")

        # include("integration/test_end_to_end.jl")
    end

    # C interface tests (compiles and runs actual C code via gcc)
    if GROUP == "All" || GROUP == "C"
        include("c_interface/test_c_api.jl")
    end
end
