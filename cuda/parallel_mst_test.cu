#include <iostream>
#include <vector>
#include <limits.h>
#include "random_graph_generator.h"

#define BLOCK_SIZE 256

typedef long long int ll;
typedef pair<long long int, long long int> Edge; // Define edge type
typedef vector<vector<Edge>> Graph; // Define graph type

__global__ void initializeDistanceVector(ll* distanceVector, int numVertices, int source) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < numVertices) {
        distanceVector[tid] = (tid == source) ? 0 : LLONG_MAX;
    }
}

__global__ void updateDistanceVector(int numVertices, int* vertices, ll* weights, ll* distanceVector, bool* mstSet) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < numVertices && !mstSet[tid]) {
        for (int i = 0; i < vertices[tid]; ++i) {
            int v = vertices[tid * numVertices + i];
            ll w = weights[tid * numVertices + i];
            if (!mstSet[v] && w < distanceVector[v]) {
                distanceVector[v] = w;
            }
        }
    }
}

__global__ void findClosestNode(int numVertices, ll* distanceVector, bool* mstSet, int* closestNode) {
    __shared__ ll s_minDist[BLOCK_SIZE];
    __shared__ int s_minIdx[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    ll minDist = LLONG_MAX;
    int minIdx = -1;

    if(idx < numVertices && !mstSet[idx]) {
        minDist = distanceVector[idx];
        minIdx = idx;
    }

    s_minDist[tid] = minDist;
    s_minIdx[tid] = minIdx;

    __syncthreads();

    // Reduction to find minimum distance and its index
    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s && s_minDist[tid + s] < s_minDist[tid]) {
            s_minDist[tid] = s_minDist[tid + s];
            s_minIdx[tid] = s_minIdx[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0) {
        closestNode[blockIdx.x] = s_minIdx[0];
    }
}

void printMST(const Graph& graph, bool* mstSet, int numVertices) {
    std::cout << "Minimum Spanning Tree (MST) Edges:\n";
    for(int i = 0; i < numVertices; ++i) {
        if(mstSet[i]) {
            for(const auto& edge : graph[i]) {
                if(mstSet[edge.first]) {
                    std::cout << "(" << i << " - " << edge.first << ") Weight: " << edge.second << "\n";
                }
            }
        }
    }
}

void printGraph(int n, const Graph& graph) {
    for (int u = 0; u < n; ++u) {
        for (const auto& neighbor : graph[u]) {
            int v = neighbor.first;
            long long int weight = neighbor.second;
            cout << u << " - " << v << " : " << weight << endl;
        }
    }
    printf("\n");
}

int main() {
    int numVertices = 10;
    int source = 0;
    Graph graph = generate(numVertices);

    printf("Generated graph:\n");
    printGraph(numVertices, graph);

    // Copy graph data to device
    int* d_vertices;
    ll* d_weights;
    cudaMalloc((void**)&d_vertices, numVertices * numVertices * sizeof(int));
    cudaMalloc((void**)&d_weights, numVertices * numVertices * sizeof(ll));

    for(int i = 0; i < numVertices; ++i) {
        for(int j = 0; j < graph[i].size(); ++j) {
            int v = graph[i][j].first;
            ll w = graph[i][j].second;
            cudaMemcpy(&d_vertices[i * numVertices + j], &v, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(&d_weights[i * numVertices + j], &w, sizeof(ll), cudaMemcpyHostToDevice);
        }
    }

    // Allocate device memory
    ll* d_distanceVector;
    bool* d_mstSet;
    int* d_closestNode;
    cudaMalloc((void**)&d_distanceVector, numVertices * sizeof(ll));
    cudaMalloc((void**)&d_mstSet, numVertices * sizeof(bool));
    cudaMalloc((void**)&d_closestNode, numVertices / BLOCK_SIZE * sizeof(int));

    // Initialize distance vector and MST set
    initializeDistanceVector<<<(numVertices + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_distanceVector, numVertices, source);
    cudaMemset(d_mstSet, 0, numVertices * sizeof(bool));

    // Main loop for MST construction
    for (int i = 0; i < numVertices - 1; ++i) {
        findClosestNode<<<(numVertices + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(numVertices, d_distanceVector, d_mstSet, d_closestNode);

        int closestNode;
        cudaMemcpy(&closestNode, d_closestNode, sizeof(int), cudaMemcpyDeviceToHost);

        // Insert closestNode into MST
        cudaMemset(&d_mstSet[closestNode], 1, sizeof(bool));

        updateDistanceVector<<<(numVertices + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(numVertices, d_vertices, d_weights, d_distanceVector, d_mstSet);
    }

    // Print the MST
    bool* mstSet = new bool[numVertices];
    cudaMemcpy(mstSet, d_mstSet, numVertices * sizeof(bool), cudaMemcpyDeviceToHost);
    printMST(graph, mstSet, numVertices);

    // Clean up
    delete[] mstSet;
    cudaFree(d_distanceVector);
    cudaFree(d_mstSet);
    cudaFree(d_vertices);
    cudaFree(d_weights);
    cudaFree(d_closestNode);

    return 0;
}