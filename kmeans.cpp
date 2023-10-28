#include <stdio.h>
#include <getopt.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "kmeans.h"
#include "kmeans_cuda.h"

using namespace std;

int main(int argc, char* argv[]) {
    // PARSE COMMAND LINE ARGUMENTS
    int opt;

    int num_cluster = 0;
    int dims = 0;
    char* inputfilename;
    int max_num_iter = 0;
    double threshold = 0.0;
    bool output = false;
    int seed = 0;
    bool gpu = false;
    bool shared_mem = false;
    bool kmeans_plus_plus = false;

    while ((opt = getopt(argc, argv, "k:d:i:m:t:cs:gfp")) != -1) {
        switch (opt) {
            case 'k':
                num_cluster = atoi(optarg);
                break;
            case 'd':
                dims = atoi(optarg);
                break;
            case 'i':
                inputfilename = optarg;
                break;
            case 'm':
                max_num_iter = atoi(optarg);
                break;
            case 't':
                threshold = atof(optarg);
                break;
            case 'c':
                output = true;
                break;
            case 's':
                seed = atoi(optarg);
                break;
            case 'g':
                gpu = true;
                break;
            case 'f':
                shared_mem = true;
                break;
            case 'p':
                kmeans_plus_plus = true;
                break;
            default:
                cerr << "Unknown option " << argv[0] << endl;
                return 1;
        }
    }

    // READ INPUT FILE
    ifstream input_file(inputfilename);
    if (!input_file.is_open()) {
        cerr << "Error in opening input file" << endl;
        return 1;
    }
    string num_points_string;
    getline(input_file, num_points_string);
    int num_points = stoi(num_points_string);
    // make points 2d array
    double** points = new double*[num_points];
    for (int i=0; i<num_points; i++) {
        points[i] = new double[dims];
    }
    for(int i=0;i<num_points;i++) {
        string point_string;
        getline(input_file, point_string);
        istringstream iss(point_string);
        // index of the point
        int point_index;
        iss >> point_index;
        point_index--;
        // read remaining numbers into the points array
        for (int j=0; j<dims; j++) {
            iss >> points[point_index][j];
        }
    }

    // INIT CENTROIDS
    // make centroids dims-dimensional array (x1, x2, ..., xdim)
    double** centroids = new double*[num_cluster]; // num_cluster x dims 
    for(int i=0; i<num_cluster; i++) {
        centroids[i] = new double[dims];
    }
    // initialize centroids
    srand(seed);
    for(int i=0;i<num_cluster;i++) {
        int point_index = (int) (rand_float() * num_points); // the index of the point that will be used for this centroid
        for(int j=0;j<dims;j++) {
            centroids[i][j] = points[point_index][j];
        }
    }

    // HOST & DEVICE POINTER ALLOCATED
    // host memory
    double** old_centroids = new double*[num_cluster];
    for(int i=0; i<num_cluster; i++) {
        old_centroids[i] = new double[dims];
    }
    int* labels = new int[num_points];

    // WRAPPER FUNCTION CALLED TO START KMEANS

    if(gpu) {
        gpu_kmeans(centroids, old_centroids, points, labels, threshold, num_cluster, dims, max_num_iter, num_points);
    } else {
        seq_kmeans(centroids, old_centroids, points, labels, threshold, num_cluster, dims, max_num_iter, num_points);
    }
    
    // PRINT OUTPUTS
    printf("FINAL:\n");
    if(output) {
        printCentroids(centroids, num_cluster, dims);
    } else {
        printClusterIds(labels, num_points);
    }
    // cleanup
    for(int i=0; i<num_points; i++) {
        delete[] points[i];
    }
    delete[] points;
    for(int i=0; i<num_cluster; i++) {
        delete[] centroids[i];
    }
    delete[] centroids;
    for(int i=0; i<num_cluster; i++) {
        delete[] old_centroids[i];
    }
    delete[] old_centroids;
    delete labels;
}