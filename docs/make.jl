using Documenter
using RunwayLib

# Generate python_interface.md from template
template = read(joinpath(@__DIR__, "src", "python_interface_template.md"), String)
example = read(joinpath(@__DIR__, "..", "test", "python_interface", "smoke_test.py"), String)
output = replace(template, "{{PYTHON_EXAMPLE}}" => example)
write(joinpath(@__DIR__, "src", "python_interface.md"), output)

makedocs(
    sitename="RunwayLib",
    format=Documenter.HTML(),
    modules=[RunwayLib],
    repo=Remotes.GitHub("RomeoV", "RunwayLib.jl"),
    pages=[
        "RunwayLib.jl: Fast Pose Estimation and Runtime Assurance for Runway Landings." => "index.md",
        "camera_model.md",
        "python_interface.md",
        "C_interface.md",
        "noise_models.md",
        "integrity_check.md",
        "api_reference.md",
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo="github.com/RomeoV/RunwayLib.jl.git"
)
