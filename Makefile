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
TEST_SRCS = $(wildcard $(TEST_DIR)/*.cpp)

# Object files
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)
UTIL_OBJS = $(UTIL_SRCS:.cpp=.o)
TEST_OBJS = $(TEST_SRCS:.cpp=.o)

# Targets
all: test

# Compile CUDA sources
$(CUDA_DIR)/%.o: $(CUDA_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile utility sources
$(UTIL_DIR)/%.o: $(UTIL_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile test sources
$(TEST_DIR)/%.o: $(TEST_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link the test executable
test: $(CUDA_OBJS) $(UTIL_OBJS) $(TEST_OBJS)
	$(CXX) $(CXXFLAGS) $(CUDA_OBJS) $(UTIL_OBJS) $(TEST_OBJS) -o test $(LDFLAGS)

# Clean up
clean:
	rm -f $(CUDA_DIR)/*.o $(UTIL_DIR)/*.o $(TEST_DIR)/*.o test

.PHONY: all clean
