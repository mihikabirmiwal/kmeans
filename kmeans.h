// ALL WRAPPER FUNCTIONS DECLARED HERE
#include <vector>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <iostream>

using namespace std;

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
int findMinDistance(double* distances, int num_clusters) {
    double minDist = DBL_MAX;
    int index = -1;
    for(int i=0;i<num_clusters;i++) {
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
void findNearestCentroids(double** points, int* labels, double** centroids, int num_points, int num_clusters, int dims) {
    // make array to store distances
    double** distances = new double*[num_points]; // num_points x num_clusters 
    for(int i=0; i<num_points; i++) {
        distances[i] = new double[num_clusters];
    }
    // calc all distances
    for(int r=0;r<num_points;r++) {
        double* point = points[r];
        for(int c=0;c<num_clusters;c++) {
            distances[r][c] = calcDistance(point, centroids[c], dims);
        }
    }
    // printf("DISTANCES\n");
    // print(distances, num_points, num_clusters);
    // take argmax based on index
    for(int i=0;i<num_points;i++) {
        labels[i] = findMinDistance(distances[i], num_clusters);
    }
    // free distances memory
    for(int i=0; i<num_points; i++) {
        delete[] distances[i];
    }
    delete[] distances;
}

// new centroids = average of all points that map to each centroid
// overrides values currently in centroids
void averageLabeledCentroids(double** points, int* labels, int num_clusters, int num_points, double** centroids, int dims) {
    // zero out all the centroid values
    for(int r=0;r<num_clusters;r++) {
        for(int c=0;c<dims;c++) {
            centroids[r][c] = 0.0;
        }
    }
    // sum up across labels, track frequencies
    int* freqs = new int[num_clusters];
    for(int i=0;i<num_clusters;i++) freqs[i] = 0;
    // printf("init freqs\n");
    // print(freqs, num_clusters);
    for(int i=0;i<num_points;i++) {
        double* point = points[i];
        int label = labels[i];
        for(int j=0;j<dims;j++) {
            centroids[label][j] += point[j];
        }
        freqs[label]++;
    }
    // printf("after summed\n");
    // print(centroids, num_clusters, dims);
    // printf("freqs\n");
    // print(freqs, num_clusters);
    // divide each centroid value by its frequency
    for(int i=0;i<num_clusters;i++) {
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
bool converged(double** centroids, double** oldCentroids, double threshold, int num_clusters, int dims) {
    // init sorted vectors
    vector<vector<double>> centroidSorted(num_clusters, vector<double>(dims));
    vector<vector<double>> centroidOldSorted(num_clusters, vector<double>(dims));
    for (int i = 0; i < num_clusters; i++) {
        for (int j = 0; j < dims; j++) {
            centroidSorted[i][j] = centroids[i][j];
            centroidOldSorted[i][j] = oldCentroids[i][j];
        }
    }
    // sort the vectors so that we are comparing the correct centroids
    sort(centroidSorted.begin(), centroidSorted.end(), compare);
    sort(centroidOldSorted.begin(), centroidOldSorted.end(), compare);
    // make sure all points are within threshold of each other
    for(int i=0;i<num_clusters;i++) {
        if(calcDistance(centroidSorted[i], centroidOldSorted[i], dims)>threshold) return false;
    }
    return true;
}

void printClusterIds(int* labels, int num_points) {
    printf("clusters:");
    for(int i=0;i<num_points;i++) {
        printf(" %d", labels[i]);
    }     
}

void printCentroids(double** centroids, int num_clusters, int dims) {
    for (int clusterId=0; clusterId<num_clusters; clusterId++) {
        printf("%d", clusterId);
        for (int d=0; d<dims; d++) {
            printf(" %lf", centroids[clusterId][d]);
        }   
        printf("\n");
    }
}