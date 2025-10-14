using Documenter
using RunwayLib

makedocs(
    sitename="RunwayLib",
    format=Documenter.HTML(),
    modules=[RunwayLib],
    repo=Remotes.GitHub("RomeoV", "RunwayLib.jl")
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
