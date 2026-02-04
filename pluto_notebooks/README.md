# RunwayLib.jl: Runway Pose Estimation

This interactive notebook demonstrates pose estimation with integrity monitoring for vision-based aircraft pose estimation.

## Prerequisites

### 1. Install Julia (version 1.10 or later, version 1.12 recommended)
- Download Julia: https://julialang.org/downloads/

- Windows users:
    ```
    winget install julia -s msstore
    ```
- Linux/macOS users:
    ```
    curl -fsSL https://install.julialang.org | sh
    ```

This will give you the `julia` command in your terminal.

> #### Already have Julia?
> - If you already have `julia` and it's not updated to version 1.12, please run the following in a terminal:
> ```
> juliaup update
> ```

### 2. Install Pluto
Open Julia (type `julia` in terminal, or open the Julia app) and run:
```julia
using Pkg
Pkg.add("Pluto")
```

### 3. Install Packages
_**Note**: You only need to do this once._
1. _Clone this git repo:_
    1. Open a terminal and navigate to where you want the code to live.
    1. Run:
        ```
        git clone --depth 1 https://github.com/RomeoV/RunwayLib.jl
        ```

## Running the Notebook

1. **Open a terminal** and navigate to the `pluto_notebooks` folder:
   ```bash
   cd path/to/RunwayLib.jl
   ```

2. **Start Julia** and launch Pluto:
   ```julia
   using Pluto
   Pluto.run()
   ```

3. **A browser window will open.** In the file path box at the top, enter:
   ```
   pluto_notebooks/full-example-notebook.jl
   ```

4. **Wait for the notebook to load** — the first run may take several minutes as Julia compiles dependencies.

## Using the Notebook

### Interactive Controls

- **Sliders**: Adjust perturbation magnitudes in different directions
- **Checkboxes**: Select which runway corners have faults (Near Left, Far Left, Far Right, Near Right)
- **Dropdown menu**: Choose which axis to analyze:
  - Position: alongtrack, crosstrack, altitude
  - Rotation: roll, pitch, yaw


## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Package not found" errors | Click the package manager icon (📦) in Pluto's sidebar |
| Slow first run | Normal — Julia compiles code on first use. Subsequent runs are faster. |
| Plots not showing | Try refreshing the browser |
| Notebook won't load | Ensure you're in the `RunwayLib.jl` directory when starting Julia |