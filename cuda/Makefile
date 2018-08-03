CXX := nvcc
TARGET := fftnet_cuda
CUDNN_PATH := /usr/local/cuda
INCLUDE_DIRS := -I$(CUDNN_PATH)/include -Isrc
LIBS := -L$(CUDNN_PATH)/lib64 -L/usr/local/lib -lcudnn
CXXFLAGS := -std=c++14 -O2 -ccbin clang-3.9 -lstdc++

all: ${TARGET}

all: ${TARGET}

fftnet_cuda: src/fftnet_main.cc
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) $(LIBS) $< -o build/${TARGET}


cudnn_context_test: cudnn_context.o exception.o cudnn_context_test.o
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) $(LIBS) -o build/$@ $(addprefix build/,$^)
	./build/$@

exception_test: exception.o exception_test.o
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) $(LIBS) -o build/$@ $(addprefix build/,$^)
	./build/$@

%.o: tests/%.cc
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c -o build/${@F} $<

%.o: src/%.cc
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c -o build/${@F} $<

.phony: clean

clean:
	rm -f build/*

