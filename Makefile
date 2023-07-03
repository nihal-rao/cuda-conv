CUDAFLAGS= -std=c++11 -c -arch=sm_61 -rdc=true

program : conv_cpu.o conv_cuda_naive.o main.o
	nvcc -std=c++11 -arch=sm_61 -rdc=true -o program main.o conv_cpu.o conv_cuda_naive.o

conv_cuda_naive.o : conv_cuda_naive.cu conv.h
	nvcc $(CUDAFLAGS) conv_cuda_naive.cu

conv_cpu.o : conv_cpu.cpp conv.h
	g++ -g -c conv_cpu.cpp

main.o : main.cu conv.h
	nvcc $(CUDAFLAGS) main.cu

clean :
	rm program conv_cpu.o conv_cuda_naive.o main.o
