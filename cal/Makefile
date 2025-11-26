# Compiler
NVCC = nvcc

# Flags
# sm_80 is for A100 (Perlmutter)
NVCC_FLAGS = -O3 -std=c++14 -gencode arch=compute_80,code=sm_80

# --- FIX IS HERE ---
# use the env variable from the module, or fallback to a default if not set
NCCL_PATH ?= $(NCCL_DIR)
MPI_PATH ?= $(MPICH_DIR)

# Includes: Add NCCL and MPI headers
INCLUDES = -I$(NCCL_PATH)/include -I$(MPI_PATH)/include

# Libraries: Add NCCL and MPI libs
# We also link -lmpi for the bootstrap code we added
LIBRARIES = -L$(NCCL_PATH)/lib -lnccl -L$(MPI_PATH)/lib -lmpi -lcudart

# Output binary
TARGET = ring_allreduce

all: $(TARGET)

$(TARGET): ring_allreduce.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) ring_allreduce.cu -o $(TARGET) $(LIBRARIES)

clean:
	rm -f $(TARGET) nccl_unique_id.bin