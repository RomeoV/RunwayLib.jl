JULIA ?= $(shell which julia)
SRC = $(shell find ../src -name "*.jl")
CC = gcc

CFLAGS = -Wall -O2
BUILD_DIR = RunwayLibCompiled2
RPATH_FLAGS = -Wl,-rpath,'$$ORIGIN/$(BUILD_DIR)/lib' \
              -Wl,-rpath,'$$ORIGIN/$(BUILD_DIR)/lib/julia'
LDFLAGS = -L$(BUILD_DIR)/lib -L$(BUILD_DIR)/lib/julia
LIBS = -lposeest -ljulia

run: mainc
	./mainc

mainc: main.c $(BUILD_DIR)/include/libposeest.h $(BUILD_DIR)/lib
	$(CC) $(CFLAGS) $(RPATH_FLAGS) -I$(BUILD_DIR)/include -o mainc main.c $(LDFLAGS) $(LIBS)

$(BUILD_DIR)/lib: $(BUILD_DIR)/lib/libposeest.so

$(BUILD_DIR)/lib/libposeest.so: loadrunwaylib.jl Project.toml ../Project.toml ../Manifest.toml $(SRC)
	-rm -rf $(BUILD_DIR)
	$(JULIA) --project juliac-script.jl $(BUILD_DIR)
	mv $(BUILD_DIR)/lib/lib.so $(BUILD_DIR)/lib/libposeest.so

$(BUILD_DIR)/include/libposeest.h: libposeest.h
	@mkdir -p $(BUILD_DIR)/include
	cp libposeest.h $(BUILD_DIR)/include

../Manifest.toml:
	julia --project=.. -e "import Pkg; Pkg.instantiate()"

.PHONY: clean run 

clean:
	-rm -rf mainc $(BUILD_DIR)
