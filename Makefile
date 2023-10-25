all:
	/lusr/cuda-11.6/bin/nvcc -o kmeans.o -c kmeans.cpp
	/lusr/cuda-11.6/bin/nvcc -o kmeans_kernel.o -c kmeans_kernel.cu
	/lusr/cuda-11.6/bin/nvcc -o kmeans kmeans.o kmeans_kernel.o

clean:
	rm kmeans.o
	rm kmeans_kernel.o
	rm kmeans

test:
	g++ -O3 -o kmeans kmeans.cpp -pthread

	rm kmeans