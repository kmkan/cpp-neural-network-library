NVCC_PATH_OVERRIDE = /usr/local/cuda-12.9/bin/
NVCC = $(NVCC_PATH_OVERRIDE)nvcc

CXX = g++

INCLUDE_DIRS = -Iinclude

CXXFLAGS = -std=c++11 -Wall -O2 $(INCLUDE_DIRS)
CUDA_ARCH = -arch=sm_75
NVCCFLAGS = -std=c++11 $(CUDA_ARCH) -O2 --compiler-options '-Wall' $(INCLUDE_DIRS)

LDFLAGS =
CUDA_LIBS = -lcudart -lcublas

TARGET = nn_cuda_test

SRC_DIR = src
OBJ_DIR = obj

CPP_SRCS =

CUDA_CPP_SRCS_NAMES = Matrix.cpp main.cpp Layer.cpp Network.cpp MNISTLoader.cpp
CUDA_CPP_SRCS = $(patsubst %,$(SRC_DIR)/%,$(CUDA_CPP_SRCS_NAMES))

CPP_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SRCS))
CUDA_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CUDA_CPP_SRCS))

all: $(TARGET)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/Matrix.o: $(SRC_DIR)/Matrix.cpp include/Matrix.h
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $(SRC_DIR)/Matrix.cpp -o $@

$(OBJ_DIR)/main.o: $(SRC_DIR)/main.cpp include/Matrix.h include/Network.h include/MNISTLoader.h
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $(SRC_DIR)/main.cpp -o $@

$(OBJ_DIR)/Layer.o: $(SRC_DIR)/Layer.cpp include/Layer.h include/Matrix.h
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $(SRC_DIR)/Layer.cpp -o $@

$(OBJ_DIR)/Network.o: $(SRC_DIR)/Network.cpp include/Network.h include/Layer.h include/Matrix.h
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $(SRC_DIR)/Network.cpp -o $@

$(OBJ_DIR)/MNISTLoader.o: $(SRC_DIR)/MNISTLoader.cpp include/MNISTLoader.h include/Matrix.h
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $(SRC_DIR)/MNISTLoader.cpp -o $@

$(TARGET): $(CPP_OBJS) $(CUDA_OBJS)
	$(NVCC) $(CUDA_ARCH) $^ -o $@ $(LDFLAGS) $(CUDA_LIBS)
	@echo "Linked successfully: $@"

clean:
	rm -f $(TARGET) $(OBJ_DIR)/*.o
	@echo "Cleaned project."
	@rmdir $(OBJ_DIR) 2>/dev/null || true

.PHONY: all clean
