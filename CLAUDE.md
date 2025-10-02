## Package Structure

- This package has three main components. The Julia code (`src`), the compiled library (`juliac`), and some python bindings code (`poseest_py`). 
- To use the Julia functionality from python, you first need to recompile the Julia code to a C-library using the Makefile in the `juliac` directory. Notice that the library is then symlinked to the python `native` code directory.
- In particular, PYTHON TESTS WILL NOT PASS IF YOU DON'T RECOMPILE THE LIBRARY after fixing an issue in the julia code.

## Testing Tips

- Note that often times you're better off running tests directly with something like `julia --project=test test/...` rather than `Pkg.test()` as it can take a long time.
- I have the bash aliases ```
alias juliaserver='julia --startup-file=no -e "using DaemonMode; serve()"'
alias juliaclient='julia --startup-file=no -e "using DaemonMode; runargs()"'
alias juliaclientexpr='julia --startup-file=no -e "using DaemonMode; runexpr(\"begin; \$(ARGS[1]); end\")" -- '
```. You can assume a server is running, and directly interact this way to run simple scripts or tests.
