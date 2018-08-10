CXX := nvcc
TARGET := fftnet_cuda
CUDNN_PATH := /usr/local/cuda
INCLUDE_DIRS := -I$(CUDNN_PATH)/include -I/usr/include/hdf5/serial -Isrc
LIBS := -L$(CUDNN_PATH)/lib64 -L/usr/local/lib -lcudnn -lcublas -lhdf5_cpp -lhdf5_serial -lm
CXXFLAGS := -std=c++14 -O2 -ccbin clang-3.9 -lstdc++

all: ${TARGET}

test: cudnn_context_test exception_test tensor_test array4d_test conv1d_test linear_test fftnet_block_test fftnet_test

fftnet_cuda: build/cudnn_context.o build/tensor.o build/kernel.o build/array4d.o build/exception.o build/conv1d.o build/linear.o build/relu.o build/fftnet_block.o build/fftnet.o build/fftnet_main.o
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) $(LIBS) -o build/${TARGET} $^
	./build/$@ /tmp/model_ckpt.h5 /tmp/lc_file.h5 /tmp/output.h5


cudnn_context_test: build/cudnn_context.o build/exception.o build/data_type.o build/catch.o build/cudnn_context_test.o
	$(CXX) $(CXXFLAGS) $(LIBS) -o build/$@ $^
	./build/$@

exception_test: build/exception.o build/catch.o build/exception_test.o
	$(CXX) $(CXXFLAGS) $(LIBS) -o build/$@ $^
	./build/$@

tensor_test: build/tensor.o build/array4d.o build/data_type.o build/exception.o build/catch.o build/tensor_test.o
	$(CXX) $(CXXFLAGS) $(LIBS) -o build/$@ $^
	./build/$@

kernel_test: build/kernel.o build/array4d.o build/data_type.o build/exception.o build/catch.o build/kernel_test.o
	$(CXX) $(CXXFLAGS) $(LIBS) -o build/$@ $^
	./build/$@

array4d_test: build/tensor.o build/kernel.o build/array4d.o build/exception.o build/catch.o build/array4d_test.o
	$(CXX) $(CXXFLAGS) $(LIBS) -o build/$@ $^
	./build/$@

conv1d_test: build/cudnn_context.o build/tensor.o build/kernel.o build/array4d.o build/data_type.o build/exception.o build/conv1d.o build/catch.o build/conv1d_test.o
	$(CXX) $(CXXFLAGS) $(LIBS) -o build/$@ $^
	./build/$@

linear_test: build/cudnn_context.o build/tensor.o build/array4d.o build/data_type.o build/exception.o build/linear.o build/catch.o build/linear_test.o
	$(CXX) $(CXXFLAGS) $(LIBS) -o build/$@ $^
	./build/$@

fftnet_block_test: build/cudnn_context.o build/tensor.o build/kernel.o build/array4d.o build/data_type.o build/exception.o build/conv1d.o build/relu.o build/fftnet_block.o build/catch.o build/fftnet_block_test.o
	$(CXX) $(CXXFLAGS) $(LIBS) -o build/$@ $^
	./build/$@

fftnet_test: build/cudnn_context.o build/tensor.o build/kernel.o build/array4d.o build/data_type.o build/exception.o build/conv1d.o build/relu.o build/linear.o build/fftnet_block.o build/fftnet.o build/catch.o build/fftnet_test.o
	$(CXX) $(CXXFLAGS) $(LIBS) -o build/$@ $^
	./build/$@

cuda_test: build/catch.o build/cuda_test.o
	$(CXX) $(CXXFLAGS) $(LIBS) -o build/$@ $^
	./build/$@

cublas_test: build/catch.o build/cublas_test.o
	$(CXX) $(CXXFLAGS) $(LIBS) -o build/$@ $^
	./build/$@

hdf5_test: build/cudnn_context.o build/data_type.o build/array4d.o build/exception.o build/tensor.o build/linear.o build/kernel.o build/conv1d.o build/catch.o build/hdf5_test.o
	$(CXX) $(CXXFLAGS) $(LIBS) -o build/$@ $^
	./build/$@

hidden_test: build/tensor.o build/exception.o build/array4d.o build/cudnn_context.o build/data_type.o build/array4d.o build/exception.o build/tensor.o build/linear.o build/kernel.o build/conv1d.o build/catch.o build/hidden_test.o
	$(CXX) $(CXXFLAGS) $(LIBS) -o build/$@ $^
	./build/$@

build/%.o: tests/%.cc
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c -o build/${@F} $<

build/%.o: src/%.cc
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c -o build/${@F} $<

.phony: clean

clean:
	rm -f build/*

