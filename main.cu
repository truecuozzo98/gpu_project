#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define CUDA_CHECK(ans)                                                         \
    {                                                                            \
        gpuAssert((ans), __FILE__, __LINE__);                                   \
    }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
                file, line);
        if (abort)
            exit(code);
    }
}

__global__ void updateWeights(int *weights, bool *inMST, int *adjacencyList,
                               int *E, int *W, int nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nodes && !inMST[tid]) {
        for (int i = 0; i < nodes; ++i) {
            if (adjacencyList[tid * nodes + i] != -1 && !inMST[i]) {
                int idx = adjacencyList[tid * nodes + i];
                if (weights[i] > W[idx]) {
                    weights[i] = W[idx];
                }
            }
        }
    }
}

__global__ void findMinWeight(int *weights, bool *inMST, int *minWeightIdx,
                              int nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nodes && !inMST[tid]) {
        if (weights[tid] < weights[*minWeightIdx]) {
            *minWeightIdx = tid;
        }
    }
}

int main() {
    int nodes, edges;

    // Read from file
    FILE *file = fopen("graph2p14", "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file.\n");
        exit(1);
    }

    fscanf(file, "%d %d", &nodes, &edges);

    // Allocate memory
    int *E = (int *)malloc(2 * edges * sizeof(int));
    int *W = (int *)malloc(2 * edges * sizeof(int));

    // Read edges and weights from file
    for (int i = 0; i < edges; ++i) {
        int node1, node2, weight;
        fscanf(file, "%d %d %d", &node1, &node2, &weight);
        E[2 * i] = node1;
        E[2 * i + 1] = node2;
        W[2 * i] = weight;
        W[2 * i + 1] = weight;
    }
    fclose(file);

    int *V = (int *)malloc(nodes * sizeof(int));
    int cumulative_sum = 0, limit;
    for (int i = 0; i < nodes; ++i) {
        V[i] = cumulative_sum;
        limit = nodes;
        for (int j = 0; j < limit; ++j) {
            if (adjacencyList[i * nodes + j] != -1) {
                adjacencyList[V[i] + j] = adjacencyList[i * nodes + j];
            }
        }
        cumulative_sum += limit;
    }

    long long int edge_sum = 0;
    int current = 0, count = 0;

    int *parent = (int *)malloc(nodes * sizeof(int));
    int *weights = (int *)malloc(nodes * sizeof(int));
    bool *inMST = (bool *)malloc(nodes * sizeof(bool));

    parent[0] = -1;
    for (int i = 0; i < nodes; ++i) {
        weights[i] = INT_MAX;
        inMST[i] = false;
    }

    int *d_weights, *d_adjacencyList, *d_E, *d_W, *d_minWeightIdx;
    bool *d_inMST;
    CUDA_CHECK(cudaMalloc((void **)&d_weights, nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_adjacencyList,
                           nodes * nodes * sizeof(int)));
    CUDA_CHECK(
        cudaMalloc((void **)&d_E, 2 * edges * sizeof(int))); // No. of edges
    CUDA_CHECK(cudaMalloc((void **)&d_W, 2 * edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_inMST, nodes * sizeof(bool)));
    CUDA_CHECK(cudaMalloc((void **)&d_minWeightIdx, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_weights, weights, nodes * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_adjacencyList, adjacencyList,
                          nodes * nodes * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_E, E, 2 * edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_W, W, 2 * edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inMST, inMST, nodes * sizeof(bool),
                          cudaMemcpyHostToDevice));

    clock_t begin = clock();

    while (count < nodes - 1) {
        ++count;
        inMST[current] = true;
        CUDA_CHECK(cudaMemcpy(d_inMST, inMST, nodes * sizeof(bool),
                              cudaMemcpyHostToDevice));
        updateWeights<<<(nodes + 255) / 256, 256>>>(
            d_weights, d_inMST, d_adjacencyList, d_E, d_W, nodes);

        int minWeightIdx = -1;
        CUDA_CHECK(cudaMemcpy(d_minWeightIdx, &minWeightIdx, sizeof(int),
                              cudaMemcpyHostToDevice));
        findMinWeight<<<(nodes + 255) / 256, 256>>>(d_weights, d_inMST,
                                                      d_minWeightIdx, nodes);
        CUDA_CHECK(cudaMemcpy(&minWeightIdx, d_minWeightIdx, sizeof(int),
                              cudaMemcpyDeviceToHost));

        printf("Min Weight Index: %d\n", minWeightIdx);

        parent[minWeightIdx] = current;
        edge_sum += weights[minWeightIdx];
        weights[minWeightIdx] = INT_MAX;
        current = minWeightIdx;
    }

    clock_t end = clock();
    printf("Sum of Edges in MST: %lld\n", edge_sum);
    double elapsed_time = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Execution time: %f\n", elapsed_time);

    free(V);
    free(E);
    free(W);
    free(parent);
    free(weights);
    free(inMST);
    free(adjacencyList);

    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_adjacencyList));
    CUDA_CHECK(cudaFree(d_E));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_inMST));
    CUDA_CHECK(cudaFree(d_minWeightIdx));

    return 0;
}
