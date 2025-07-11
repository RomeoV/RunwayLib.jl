using Test
using RunwayPoseEstimation

@testset "RunwayPoseEstimation.jl" begin
    include("unit/test_coordinate_systems.jl")
    include("unit/test_camera_model.jl")
    include("unit/test_data_management.jl")
    include("unit/test_pose_estimation.jl")
    include("unit/test_uncertainty_quantification.jl")
    include("unit/test_integrity_monitoring.jl")
    include("unit/test_visualization.jl")
    
    include("integration/test_end_to_end.jl")
end
