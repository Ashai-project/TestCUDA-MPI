# Compiler and flags
CXX = gcc
NVCC = nvcc
CXXFLAGS = -I./src -I./src/util -I./src/cuda
NVCCFLAGS = -I./src -I./src/cuda -lmpi
LDFLAGS = -lcuda -lcudart

# Directories
SRC_DIR = src
CUDA_DIR = $(SRC_DIR)/cuda
UTIL_DIR = $(SRC_DIR)/util
TEST_DIR = $(SRC_DIR)/test

# Source files
CUDA_SRCS = $(wildcard $(CUDA_DIR)/*.cu)
UTIL_SRCS = $(wildcard $(UTIL_DIR)/*.cpp)
TEST_CPP_SRCS = $(wildcard $(TEST_DIR)/*.cpp)
TEST_CUDA_SRCS = $(wildcard $(TEST_DIR)/*.cu)

# Object files
CUDA_OBJS = $(CUDA_SRCS:$(CUDA_DIR)/%.cu=$(CUDA_DIR)/%.o)
UTIL_OBJS = $(UTIL_SRCS:$(UTIL_DIR)/%.cpp=$(UTIL_DIR)/%.o)
TEST_CPP_OBJS = $(TEST_CPP_SRCS:$(TEST_DIR)/%.cpp=$(TEST_DIR)/%.o)
TEST_CUDA_OBJS = $(TEST_CUDA_SRCS:$(TEST_DIR)/%.cu=$(TEST_DIR)/%.o)

# All object files
ALL_OBJS = $(CUDA_OBJS) $(UTIL_OBJS) $(TEST_CPP_OBJS) $(TEST_CUDA_OBJS)

# Targets
all: test_executable

# Compile CUDA sources
$(CUDA_DIR)/%.o: $(CUDA_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile utility sources
$(UTIL_DIR)/%.o: $(UTIL_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile test C++ sources
$(TEST_DIR)/%.o: $(TEST_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile test CUDA sources
$(TEST_DIR)/%.o: $(TEST_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Link the test executable
test_executable: $(ALL_OBJS)
	$(NVCC) $(ALL_OBJS) -o $(TEST_DIR)/test $(LDFLAGS)

# Clean up
clean:
	rm -f $(CUDA_OBJS) $(UTIL_OBJS) $(TEST_CPP_OBJS) $(TEST_CUDA_OBJS) $(TEST_DIR)/test

.PHONY: all clean
