CXX := nvcc
TARGET := fftnet_cuda
CUDNN_PATH := /usr/local/cuda
INCLUDE_DIRS := -I$(CUDNN_PATH)/include -Isrc
LIBS := -L$(CUDNN_PATH)/lib64 -L/usr/local/lib -lcudnn
CXXFLAGS := -std=c++14 -O2 -ccbin clang-3.9 -lstdc++

all: ${TARGET}

test: cudnn_context_test exception_test tensor_test

fftnet_cuda: build/fftnet_main.o
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) $(LIBS) -o build/${TARGET} $^


cudnn_context_test: build/cudnn_context.o build/exception.o build/data_type.o build/catch.o build/cudnn_context_test.o
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) $(LIBS) -o build/$@ $^
	./build/$@

exception_test: build/exception.o build/catch.o build/exception_test.o
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) $(LIBS) -o build/$@ $^
	./build/$@

tensor_test: build/tensor.o build/data_type.o build/exception.o build/catch.o build/tensor_test.o
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) $(LIBS) -o build/$@ $^
	./build/$@

kernel_test: build/kernel.o build/data_type.o build/exception.o build/catch.o build/kernel_test.o
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) $(LIBS) -o build/$@ $^
	./build/$@

convolution_test: build/tensor.o build/data_type.o build/exception.o build/convolution.o build/catch.o build/convolution_test.o
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) $(LIBS) -o build/$@ $(addprefix build/,$^)
	./build/$@

build/%.o: tests/%.cc
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c -o build/${@F} $<

build/%.o: src/%.cc
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c -o build/${@F} $<

.phony: clean

clean:
	rm -f build/*

