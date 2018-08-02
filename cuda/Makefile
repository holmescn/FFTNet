CXX := nvcc
TARGET := fftnet_cuda
CUDNN_PATH := /usr/local/cuda
HEADERS := -I $(CUDNN_PATH)/include
LIBS := -L $(CUDNN_PATH)/lib64 -L/usr/local/lib -lcudnn
CXXFLAGS := -std=c++14 -O2 -ccbin clang-3.9 -lstdc++

all: ${TARGET}

all: ${TARGET}

fftnet_cuda: fftnet_main.cu
	$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) $< -o build/${TARGET}

.phony: clean

clean:
	rm -f build/${TARGET}

