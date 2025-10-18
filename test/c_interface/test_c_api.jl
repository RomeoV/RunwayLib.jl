using Test
using JSON

@testset "C API Interface" begin
    # Get repository root
    repo_root = dirname(dirname(@__DIR__))
    juliac_dir = joinpath(repo_root, "juliac")
    test_c_dir = joinpath(repo_root, "test", "c_interface")

    @testset "Build library" begin
        # Run make in juliac directory to build the library
        cd(juliac_dir) do
            result = run(`make`)
            @test result.exitcode == 0 || success(result)
        end
    end

    @testset "Compile and run test" begin
        # Compile the test C file
        test_exe = joinpath(test_c_dir, "test_c_api")
        compiled_lib = joinpath(juliac_dir, "RunwayLibCompiled")

        # Build compilation command
        gcc_cmd = [
            "gcc", "-Wall", "-O2", "-std=c99",
            "-I$(joinpath(compiled_lib, "include"))",
            "-o", test_exe,
            joinpath(test_c_dir, "test_main.c"),
            "-L$(joinpath(compiled_lib, "lib"))",
            "-L$(joinpath(compiled_lib, "lib", "julia"))",
            "-lposeest", "-ljulia",
            "-Wl,-rpath,$(joinpath(compiled_lib, "lib"))",
            "-Wl,-rpath,$(joinpath(compiled_lib, "lib", "julia"))"
        ]

        @test success(run(Cmd(gcc_cmd)))

        # Run the test executable and capture output
        output = read(`$test_exe`, String)

        # Parse JSON output
        result = JSON.parse(output)

        # Test expected values (these are the known outputs from the test case)
        @test haskey(result, "position")
        @test haskey(result, "rotation")
        @test haskey(result, "residual_norm")
        @test haskey(result, "converged")

        # Check position (approximately -2000, 12, 150)
        @test result["position"]["x"] ≈ -2000.0 atol=1.0
        @test result["position"]["y"] ≈ 12.0 atol=1.0
        @test result["position"]["z"] ≈ 150.0 atol=1.0

        # Check rotation exists and matches expected values (from docs/src/index.md)
        # Expected: yaw=0°, pitch=5°, roll=1.5°
        @test haskey(result["rotation"], "yaw")
        @test haskey(result["rotation"], "pitch")
        @test haskey(result["rotation"], "roll")
        @test result["rotation"]["yaw"] ≈ deg2rad(0.0) atol=0.01
        @test result["rotation"]["pitch"] ≈ deg2rad(5.0) atol=0.01
        @test result["rotation"]["roll"] ≈ deg2rad(1.5) atol=0.01

        # Check convergence
        @test result["converged"] == true

        # Check residual is small
        @test result["residual_norm"] < 1e-3
    end

    @testset "3DOF pose estimation" begin
        # Compile the 3dof test C file
        test_exe_3dof = joinpath(test_c_dir, "test_3dof")
        compiled_lib = joinpath(juliac_dir, "RunwayLibCompiled")

        # Build compilation command
        gcc_cmd = [
            "gcc", "-Wall", "-O2", "-std=c99",
            "-I$(joinpath(compiled_lib, "include"))",
            "-o", test_exe_3dof,
            joinpath(test_c_dir, "test_3dof.c"),
            "-L$(joinpath(compiled_lib, "lib"))",
            "-L$(joinpath(compiled_lib, "lib", "julia"))",
            "-lposeest", "-ljulia",
            "-Wl,-rpath,$(joinpath(compiled_lib, "lib"))",
            "-Wl,-rpath,$(joinpath(compiled_lib, "lib", "julia"))"
        ]

        @test success(run(Cmd(gcc_cmd)))

        # Run the test executable and capture output
        output = read(`$test_exe_3dof`, String)

        # Parse JSON output
        result = JSON.parse(output)

        # Test expected values
        @test haskey(result, "position")
        @test haskey(result, "rotation")
        @test haskey(result, "residual_norm")
        @test haskey(result, "converged")

        # Check position (should be close to 6dof result: -2000, 12, 150)
        @test result["position"]["x"] ≈ -2000.0 atol=1.0
        @test result["position"]["y"] ≈ 12.0 atol=1.0
        @test result["position"]["z"] ≈ 150.0 atol=1.0

        # Check rotation matches the known rotation we provided (from docs/src/index.md)
        # Expected: yaw=0°, pitch=5°, roll=1.5°
        @test result["rotation"]["yaw"] ≈ deg2rad(0.0) atol=0.001
        @test result["rotation"]["pitch"] ≈ deg2rad(5.0) atol=0.001
        @test result["rotation"]["roll"] ≈ deg2rad(1.5) atol=0.001

        # Check convergence
        @test result["converged"] == true

        # Check residual is small
        @test result["residual_norm"] < 1e-3
    end
end
