// ALL WRAPPER FUNCTIONS DECLARED HERE
#pragma once
#ifndef KMEANS_FUNCTIONS_H
#define KMEANS_FUNCTIONS_H

#include <vector>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <iostream>
#include <chrono>

using namespace std;

void printClusterIds(int* labels, int num_points) {
    printf("clusters:");
    for(int i=0;i<num_points;i++) {
        printf(" %d", labels[i]);
    }     
}

void printCentroids(double** centroids, int num_cluster, int dims) {
    for (int clusterId=0; clusterId<num_cluster; clusterId++) {
        printf("%d", clusterId);
        for (int d=0; d<dims; d++) {
            printf(" %lf", centroids[clusterId][d]);
        }   
        printf("\n");
    }
}

void print(double** arr, int rows, int cols) {
    for(int r=0;r<rows;r++) {
        for(int c=0;c<cols;c++) {
            printf("%lf ", arr[r][c]);
        }
        printf("\n");
    }
}

void print(int* arr, int len) {
    for(int i=0;i<len;i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

// returns a float in [0.0, 1.0)
float rand_float() {
    return static_cast<float> (rand()) / static_cast<float> ((long long) RAND_MAX+1);
}

// returns index of minimum distance in the array (will be the cluster number)
int findMinDistance(double* distances, int num_cluster) {
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

// returns the double distance between 2 dims-dimensional points
double calcDistance(double* p1, double* p2, int dims) {
    double sum = 0.0;
    for(int i=0;i<dims;i++) {
        sum += pow(p1[i]-p2[i], 2);
    }
    return sqrt(sum);
}

// overrides values currently in labels
void findNearestCentroids(double** points, int* labels, double** centroids, int num_points, int num_cluster, int dims) {
    // make array to store distances
    double** distances = new double*[num_points]; // num_points x num_cluster
    for(int i=0; i<num_points; i++) {
        distances[i] = new double[num_cluster];
    }
    // calc all distances
    for(int r=0;r<num_points;r++) {
        for(int c=0;c<num_cluster;c++) {
            distances[r][c] = calcDistance(points[r], centroids[c], dims);
        }
    }
    // take argmax based on index
    for(int i=0;i<num_points;i++) {
        labels[i] = findMinDistance(distances[i], num_cluster);
    }
    // free distances memory
    for(int i=0; i<num_points; i++) {
        delete[] distances[i];
    }
    delete[] distances;
}

// new centroids = average of all points that map to each centroid
// overrides values currently in centroids
void averageLabeledCentroids(double** points, int* labels, int num_cluster, int num_points, double** centroids, int dims) {
    // zero out all the centroid values
    for(int r=0;r<num_cluster;r++) {
        for(int c=0;c<dims;c++) {
            centroids[r][c] = 0.0;
        }
    }
    // sum up across labels, track frequencies
    int* freqs = new int[num_cluster];
    for(int i=0;i<num_cluster;i++) freqs[i] = 0;
    for(int i=0;i<num_points;i++) {
        for(int j=0;j<dims;j++) {
            centroids[labels[i]][j] += points[i][j];
        }
        freqs[labels[i]]++;
    }
    // divide each centroid value by its frequency
    for(int i=0;i<num_cluster;i++) {
        double* centroid = centroids[i];
        int freq = freqs[i];
        for(int j=0;j<dims;j++) {
            centroid[j] /= freq;
        }
    }
    delete freqs;
}

bool compare(const vector<double> &a, const vector<double> &b) {
    int dims = a.size();
    for(int dim=0;dim<dims;dim++) {
        if(a[dim]!=b[dim]) return a[dim] < b[dim];
    }
    return true;
}

// returns the double distance between 2 dims-dimensional points
double calcDistance(const vector<double> &p1, const vector<double> &p2, int dims) {
    double sum = 0.0;
    for(int i=0;i<dims;i++) {
        sum += pow(p1[i]-p2[i], 2);
    }
    return sqrt(sum);
}

// returns true if the centroids have converged (difference between old and new centroid is below a threshold)
bool converged(double** centroids, double** oldCentroids, double threshold, int num_cluster, int dims) {
    for(int i=0;i<num_cluster;i++) {
        if(calcDistance(centroids[i], oldCentroids[i], dims)>threshold) return false;
    }
    return true;
}

// runs k means sequential algorithm. labels & centroids arrays point to the final labels & centroids after algorithm is run
float* seq_kmeans(double** centroids, double** old_centroids, double** points, int* labels, double threshold, int num_cluster, int dims, int max_num_iter, int num_points) {
    auto start = chrono::high_resolution_clock::now();
    int iteration = 0;
    bool done = iteration >= max_num_iter || converged(centroids, old_centroids, threshold, num_cluster, dims);
    while(!done) {
        for(int r=0;r<num_cluster;r++) {
            for(int c=0;c<dims;c++) {
                old_centroids[r][c] = centroids[r][c];
            }
        }
        iteration++;
        findNearestCentroids(points, labels, centroids, num_points, num_cluster, dims);
        averageLabeledCentroids(points, labels, num_cluster, num_points, centroids, dims);
        done = iteration >= max_num_iter || converged(centroids, old_centroids, threshold, num_cluster, dims);
    }
    auto end = chrono::high_resolution_clock::now();
    float* times = new float[2];
    times[0] = 0.0;
    times[1] = chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    return times;
}

#endif