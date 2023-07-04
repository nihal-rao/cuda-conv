CUDAFLAGS= -std=c++11 -arch=sm_61 -rdc=true
BUILD=build

program : $(BUILD)/conv_cpu.o $(BUILD)/conv_cuda_tiling.o $(BUILD)/conv_cuda_naive.o main.cu conv.h
	nvcc $(CUDAFLAGS) main.cu $(BUILD)/conv_cpu.o $(BUILD)/conv_cuda_tiling.o $(BUILD)/conv_cuda_naive.o -o $@

$(BUILD)/conv_cuda_tiling.o : conv_cuda_tiling.cu conv.h build
	nvcc $(CUDAFLAGS) -c conv_cuda_tiling.cu -o $@

$(BUILD)/conv_cuda_naive.o : conv_cuda_naive.cu conv.h build
	nvcc $(CUDAFLAGS) -c conv_cuda_naive.cu -o $@

$(BUILD)/conv_cpu.o : conv_cpu.cpp conv.h build
	g++ -c conv_cpu.cpp -o $@ 

build:
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD) program
