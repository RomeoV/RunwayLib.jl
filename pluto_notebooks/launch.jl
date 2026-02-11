#!/usr/bin/env julia
# Launch Pluto with local RunwayLib packages pre-resolved.
# Usage: julia pluto_notebooks/launch.jl [notebook.jl]

import Pkg
Pkg.activate(; temp=true)
Pkg.add("Pluto")
using Pluto

const REPO_ROOT = dirname(@__DIR__)
const NOTEBOOKS_DIR = @__DIR__
const DEFAULT_NOTEBOOK = joinpath(NOTEBOOKS_DIR, "protection-levels.jl")

notebook = length(ARGS) >= 1 ? joinpath(NOTEBOOKS_DIR, ARGS[1]) : DEFAULT_NOTEBOOK

# Pre-resolve local packages into the notebook's environment
Pluto.activate_notebook_environment(notebook)
Pkg.develop(; path=REPO_ROOT)
Pkg.develop(; path=joinpath(REPO_ROOT, "libs", "RunwayLibProtectionLevels"))

Pluto.run(; notebook)
