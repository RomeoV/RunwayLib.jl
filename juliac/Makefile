JULIA ?= $(shell which julia)
SRCDIR = ../src
SRC = $(shell find $(SRCDIR) -name "*.jl")
CC = gcc

CFLAGS = -Wall -O2
COMPILED_DIR = RunwayLibCompiled
LIB_DIR = $(COMPILED_DIR)/lib
INCLUDE_DIR = $(COMPILED_DIR)/include
SHARE_DIR = $(COMPILED_DIR)/share
RPATH_FLAGS = -Wl,-rpath,\$$ORIGIN/$(LIB_DIR) -Wl,-rpath,\$$ORIGIN/$(LIB_DIR)/julia
LDFLAGS = -L$(LIB_DIR) -L$(LIB_DIR)/julia
LIBS = -lposeest -ljulia

run: mainc bundle
	JULIA_DEPOT_PATH=$(SHARE_DIR)/julia ./mainc

mainc: main.c $(INCLUDE_DIR)/libposeest.h $(LIB_DIR)
	$(CC) $(CFLAGS) $(RPATH_FLAGS) -I$(INCLUDE_DIR) -o mainc main.c $(LDFLAGS) $(LIBS)

$(LIB_DIR)/libposeest.so: loadrunwaylib.jl Project.toml Manifest.toml $(SRC)
	@mkdir -p $(LIB_DIR)
	JULIAC=$$(julia -e 'print(normpath(joinpath(Sys.BINDIR, Base.DATAROOTDIR, "julia", "juliac", "juliac.jl")))' 2>/dev/null || echo "") ; \
	$(JULIA) --project=. --depwarn=error "$$JULIAC" --experimental --trim=unsafe-warn --compile-ccallable --output-lib $(LIB_DIR)/libposeest loadrunwaylib.jl --relative-rpath

Manifest.toml: Project.toml ../Project.toml
	-rm -f Manifest.toml
	$(JULIA) --project=. -e 'using Pkg; Pkg.instantiate()'
	@touch $@ # Pkg.instantiate doesn't update the mtime if there are no changes

# Create include directory with header file
$(INCLUDE_DIR)/libposeest.h: libposeest.h
	@mkdir -p $(INCLUDE_DIR)
	cp libposeest.h $(INCLUDE_DIR)/

$(LIB_DIR): $(LIB_DIR)/.installed $(LIB_DIR)/libposeest.so

# Complete bundle target
bundle: $(LIB_DIR) $(LIB_DIR)/.installed $(INCLUDE_DIR)/libposeest.h $(LIB_DIR)/libposeest.so
	@echo "RunwayLibCompiled bundle created successfully"
	@echo "Bundle size: $$(du -sh $(COMPILED_DIR) | cut -f1)"

$(LIB_DIR)/.installed: Manifest.toml
	@echo "Creating directory structure..."
	@mkdir -p $(LIB_DIR)
	-rm -rf $(SHARE_DIR)
	-rm -rf $(LIB_DIR)/julia
	-rm -f $(LIB_DIR)/libjulia*
	@echo "Copying Julia libraries..."
	$(JULIA) --project=@PackageCompiler -e "isnothing(Base.find_package(\"PackageCompiler\")) && (import Pkg; Pkg.add(\"PackageCompiler\"))"
	COMPILED_DIR=$(COMPILED_DIR) $(JULIA) --project=@PackageCompiler ./install_libraries_and_artifacts.jl
	@echo "Size after cleaning: $$(du -sh $(COMPILED_DIR) | cut -f1)"
	@touch $(LIB_DIR)/.installed
	@echo "Julia libraries installed to $(LIB_DIR)"

.PHONY: clean run bundle

clean:
	-rm -rf mainc Manifest.toml abiinfo.json $(COMPILED_DIR)
