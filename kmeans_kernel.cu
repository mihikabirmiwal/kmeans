#include <iostream>
#include <cfloat>
#include <cstdio>
#include <vector>
#include <cmath>

// returns the double distance between 2 dims-dimensional points
__device__ double calcDistanceCuda(double* p1, double* p2, int dims) {
    double sum = 0.0;
    for(int i=0;i<dims;i++) {
        sum += pow(p1[i]-p2[i], 2);
    }
    return sqrt(sum);
}

// returns index of minimum distance in the array (will be the cluster number)
__device__ int findMinDistanceCuda(double* distances, int num_cluster) {
    double minDist = DBL_MAX;
    int index = -1;
    for(int i=0;i<num_cluster;i++) {
        double distance = distances[i];
        if(distance<minDist) {
            minDist = distance;
            index = i;
        }
    }
    return index;
}

// d_distances: num_points x num_cluster [flattened]
// will be called with <<<num_points, num_cluster>>>
__global__ void calcDistances(double* d_distances, double* d_points, double* d_centroids, int dims, int num_cluster) {
    d_distances[blockIdx.x*num_cluster+threadIdx.x] = calcDistanceCuda(&d_points[blockIdx.x*dims], &d_centroids[threadIdx.x*dims], dims);
}

// take argmax based on index
// iterates over num_points values
__global__ void updateLabels(int* d_labels, double* d_distances, int num_cluster, int num_points) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index<num_points) {
        d_labels[index] = findMinDistanceCuda(&d_distances[index*num_cluster], num_cluster);
    }
}

void findNearestCentroidsCuda(double* d_points, int* d_labels, double* d_centroids, int num_points, int num_cluster, int dims) {
    // malloc distances array
    double* d_distances;
    cudaMalloc((void**)&d_distances, num_points * num_cluster * sizeof(double));

    // calc all distances (writing to d_distances)
    calcDistances<<<num_points, num_cluster>>>(d_distances, d_points, d_centroids, dims, num_cluster);    
    cudaDeviceSynchronize();

    // take argmax based on index (reading from d_distances)
    updateLabels<<<(num_points+32-1)/32, 32>>>(d_labels, d_distances, num_cluster, num_points);
    cudaDeviceSynchronize();

    // free distances memory
    cudaFree(d_distances);
}

// found this implementation online, this is not my function!
__device__ double doubleAtomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

// will be called with <<<num_points, dims>>>
__global__ void sumPointsAcrossLabels(int* d_labels, double* d_centroids, double* d_points, int dims) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    int label = d_labels[row];
    doubleAtomicAdd(&d_centroids[label*dims+col], d_points[row*dims+col]);
}

// NOTE: will be called with <<<1, num_points>>>
// have num_points things going
__global__ void sumLabelFreqs(int* d_labels, int* d_freqs, int num_points) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < num_points) {
        int label = d_labels[index];
        atomicAdd(&d_freqs[label], 1);
    }
}

// will be called with <<<1, num_points>>> [mathy]
// shared memory section: num_points*sizeof(int)
__global__ void sumLabelFreqsShmem(int* d_labels, int* d_freqs, int num_points) {
    extern __shared__ int s_labels[];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;
    if(index < num_points) {
        // copy over data to shared memory
        s_labels[lid] = d_labels[index];
        __syncthreads();

        int label = s_labels[lid];
        atomicAdd(&d_freqs[label], 1);
    }
}

// will be called with <<<num_cluster, dims>>>
__global__ void divideAcrossLabels(double* d_centroids, int* d_freqs, int dims) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    d_centroids[row*dims+col] /= d_freqs[row];
}

void averageLabeledCentroidsCuda(double* d_points, int* d_labels, int num_cluster, int num_points, double* d_centroids, int dims, bool shared) {
    // zero out all the centroid values
    cudaMemset(d_centroids, 0.0, num_cluster * dims * sizeof(double));
    cudaDeviceSynchronize();
    
    // sum up across labels, track frequencies
    int *d_freqs;
    cudaMalloc((void**)&d_freqs, num_cluster * sizeof(int));
    cudaMemset(d_freqs, 0, num_cluster * sizeof(int));
    cudaDeviceSynchronize();
    
    sumPointsAcrossLabels<<<num_points, dims>>>(d_labels, d_centroids, d_points, dims);
    if(shared) {
        sumLabelFreqsShmem<<<(num_points+32-1)/32, 32, num_points*sizeof(int)>>>(d_labels, d_freqs, num_points);
    } else {
        sumLabelFreqs<<<(num_points+32-1)/32, 32>>>(d_labels, d_freqs, num_points);
    }
    
    cudaDeviceSynchronize();

    // divide each centroid value by its frequency
    divideAcrossLabels<<<num_cluster, dims>>>(d_centroids, d_freqs, dims);
    cudaDeviceSynchronize();
    cudaFree(d_freqs);
}

// will be called with <<<1, num_cluster>>>
// arr[0] will contain the number of distances above the threshold
__global__ void checkDistanceAboveThresh(int* arr, double threshold, int dims, int num_cluster, double* d_centroids, double* d_oldCentroids) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < num_cluster) {
        if(calcDistanceCuda(&d_centroids[index*dims], &d_oldCentroids[index*dims], dims)>threshold) atomicAdd(&arr[0], 1);
    }
}

bool convergedCuda(double* d_centroids, double* d_oldCentroids, double threshold, int num_cluster, int dims) {
    // memory needed
    int *d_arr;
    cudaMalloc((void**)&d_arr, sizeof(int));
    int arr;

    // check distance, increment d_arr
    checkDistanceAboveThresh<<<1, num_cluster>>>(d_arr, threshold, dims, num_cluster, d_centroids, d_oldCentroids);
    cudaDeviceSynchronize();

    // copy back d_arr value to host 
    cudaMemcpy(&arr, d_arr, sizeof(int), cudaMemcpyDeviceToHost);
    bool hasConverged = arr == 0;

    // cleanup + return
    cudaFree(d_arr);
    return hasConverged;
}

// returns float*, [0]: mem_overhead_time, [1]: algo_time
float* gpu_kmeans(double** centroids, double** old_centroids, double** points, int* labels, double threshold, int num_cluster, int dims, int max_num_iter, int num_points, bool shared) {
    // time measurement
    float temp = 0;
    cudaEvent_t mem_overhead_start, mem_overhead_stop;
    cudaEventCreate(&mem_overhead_start);
    cudaEventCreate(&mem_overhead_stop);
    float mem_overhead_time = 0;
    cudaEvent_t algo_start, algo_stop;
    cudaEventCreate(&algo_start);
    cudaEventCreate(&algo_stop);
    float algo_time = 0;

    // allocate device memory & copy over data
    cudaEventRecord(mem_overhead_start);
    double *d_points, *d_centroids, *d_old_centroids;
    int *d_labels;

    cudaMalloc((void**)&d_points, num_points * dims * sizeof(double));
    for(int i = 0; i < num_points; i++) {
        cudaMemcpy(&d_points[i*dims], points[i], dims * sizeof(double), cudaMemcpyHostToDevice);
    }
    cudaMalloc((void**)&d_centroids, num_cluster * dims * sizeof(double));
    for(int i = 0; i < num_cluster; i++) {
        cudaMemcpy(&d_centroids[i*dims], centroids[i], dims * sizeof(double), cudaMemcpyHostToDevice);
    }
    cudaMalloc((void**)&d_old_centroids, num_cluster * dims * sizeof(double));
    for(int i = 0; i < num_cluster; i++) {
        cudaMemcpy(&d_old_centroids[i*dims], old_centroids[i], dims * sizeof(double), cudaMemcpyHostToDevice);
    }
    cudaMalloc((void**)&d_labels, num_points * sizeof(int));
    cudaMemcpy(d_labels, labels, num_points * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(mem_overhead_stop);
    cudaEventSynchronize(mem_overhead_stop);
    cudaEventElapsedTime(&temp, mem_overhead_start, mem_overhead_stop);
    mem_overhead_time += temp;

    // run loop
    cudaEventRecord(algo_start);
    int iteration = 0;
    bool done = iteration >= max_num_iter || convergedCuda(d_centroids, d_old_centroids, threshold, num_cluster, dims);
    while(!done) { 
        cudaMemcpy(d_old_centroids, d_centroids, num_cluster * dims * sizeof(double), cudaMemcpyDeviceToDevice);
        iteration++;
        findNearestCentroidsCuda(d_points, d_labels, d_centroids, num_points, num_cluster, dims);
        averageLabeledCentroidsCuda(d_points, d_labels, num_cluster, num_points, d_centroids, dims, shared);
        done = iteration >= max_num_iter || convergedCuda(d_centroids, d_old_centroids, threshold, num_cluster, dims);
    }
    cudaEventRecord(algo_stop);
    cudaEventSynchronize(algo_stop);
    cudaEventElapsedTime(&temp, algo_start, algo_stop);
    algo_time = temp;

    cudaEventRecord(mem_overhead_start);
    // copy over data back to host memory
    for(int i = 0; i < num_points; i++) {
        cudaMemcpy(points[i], &d_points[i*dims], dims * sizeof(double), cudaMemcpyDeviceToHost);
    }
    for(int i = 0; i < num_cluster; i++) {
        cudaMemcpy(centroids[i], &d_centroids[i*dims], dims * sizeof(double), cudaMemcpyDeviceToHost);
    }
    for(int i = 0; i < num_cluster; i++) {
        cudaMemcpy(old_centroids[i], &d_old_centroids[i*dims], dims * sizeof(double), cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(labels, d_labels, num_points * sizeof(int), cudaMemcpyDeviceToHost);

    // free all device memory
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_old_centroids);
    cudaFree(d_labels);
    cudaEventRecord(mem_overhead_stop);
    cudaEventSynchronize(mem_overhead_stop);
    cudaEventElapsedTime(&temp, mem_overhead_start, mem_overhead_stop);
    mem_overhead_time += temp;
    
    // return times
    float* times = new float[2];
    times[0] = mem_overhead_time;
    times[1] = algo_time;
    return times;
}

// return random floating-point value in [0.0, 1.0)
float rand_float_cuda() {
    return static_cast<float>(rand()) / static_cast<float> ((long long) RAND_MAX+1);
}

// calculates distance of every point to the closest centroid that has already been picked, stores in d_D array
// will be operating once per point
__global__ void kmeansplusplus_kernel(int num_pts, double* d_D, int* d_centroid_indices, double* d_points, int alr_selected, int dims) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index<num_pts) {
        double shortest_dist = DBL_MAX;
        for(int i=0;i<alr_selected;i++) {
            int centroid_index = d_centroid_indices[i];
            double distance = calcDistanceCuda(&d_points[centroid_index*dims], &d_points[index*dims], dims);
            if(distance<shortest_dist) {
                shortest_dist = distance;
            }
        }
        d_D[index] = shortest_dist;
    }
}

int* kmeansplusplus_init_centroids(int num_cluster, double** points, int num_pts, int dims) {
    int* centroid_indices = new int[num_cluster];
    centroid_indices[0] = (int) (rand_float_cuda()*num_pts); // stores indices we have picked so far
    double* D = new double[num_pts]; // stores distances
    int index = 1;

    double* d_points;
    double* d_D;
    int* d_centroid_indices;

    cudaMalloc((void**) &d_points, num_pts * dims * sizeof(double));
    cudaMalloc((void**) &d_D, num_pts * sizeof(double));
    cudaMalloc((void**) &d_centroid_indices, num_cluster * sizeof(int));

    for(int i=0; i<num_pts; i++) {
        cudaMemcpy(&d_points[i*dims], points[i], dims * sizeof(double), cudaMemcpyHostToDevice);
    }

    while (index < num_cluster) {
        // Here, you`ll need to compute D(x) for all points.
        cudaMemcpy(d_centroid_indices, centroid_indices, num_cluster * sizeof(int), cudaMemcpyHostToDevice);
        kmeansplusplus_kernel<<<(num_pts+32-1)/32, 32>>>(num_pts, d_D, d_centroid_indices, d_points, index, dims);
        cudaDeviceSynchronize();
        cudaMemcpy(D, d_D, num_pts * sizeof(double), cudaMemcpyDeviceToHost);

        // Choose a new initial centroid
        float total_dist = 0.0;
        for (int i = 0; i < num_pts; i++) {
            total_dist += D[i]*D[i];
        }
        float target = rand_float_cuda() * total_dist;
        float dist = 0.0;
        for (int i = 0; i < num_pts; i++) {
            dist += D[i]*D[i];
            if (target < dist) {
                centroid_indices[index] = i;
                index++;
                break;
            }
        }
    }

    // free all memory
    cudaFree(d_points);
    cudaFree(d_D);
    cudaFree(d_centroid_indices);
    delete[] D;

    return centroid_indices;
}
