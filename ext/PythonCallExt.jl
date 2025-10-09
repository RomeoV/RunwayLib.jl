module PythonCallExt
using RunwayLib
import RunwayLib: estimatepose3dof, estimatepose6dof, AbstractCameraConfig, CAMERA_CONFIG_OFFSET, _defaultnoisemodel


function estimatepose6dof(
    runway_corners::AbstractVector,
    observed_corners::AbstractVector,
    camconfig::AbstractCameraConfig{S}=CAMERA_CONFIG_OFFSET,
    noise_model::N=_defaultnoisemodel(observed_corners);
    kwargs...
) where {S,N}
    estimatepose6dof(
        map(identity, runway_corners),
        map(identity, observed_corners),
        camconfig,
        noise_model;
        kwargs...
    )
end

function estimatepose3dof(
    runway_corners::AbstractVector,
    observed_corners::AbstractVector,
    camconfig::AbstractCameraConfig{S}=CAMERA_CONFIG_OFFSET,
    noise_model::N=_defaultnoisemodel(observed_corners);
    kwargs...
) where {S,N}
    estimatepose6dof(
        map(identity, runway_corners),
        map(identity, observed_corners),
        camconfig,
        noise_model;
        kwargs...
    )
end
end
