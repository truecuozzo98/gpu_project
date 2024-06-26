#include <random>
#include <vector>
#include <limits>
#include <chrono>
#include "random_graph_generator.h"
#include "sequential_mst.h"
#define INF INT_MAX
#define MIN_EDGE_WEIGHT 10
#define MAX_EDGE_WEIGHT 100
#define MAX_NODES 100000000
#define BLOCK_SIZE 1024
#define NODES 1024
using namespace std;
typedef pair<int, int> Edge; // Define edge type
typedef vector<vector<Edge>> Graph; // Define graph type

__global__ void local_closest_node(const int *d_distance_vector, int *d_min_weights, int *d_min_nodes,
                                   const bool *d_present_in_mst) {
    __shared__ int min_weight[NODES];
    __shared__ int closest_node[NODES];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    min_weight[tid] = d_distance_vector[index];
    closest_node[tid] = index;
    __syncthreads();

    // Ignore the elements that are already present in MST
    if (d_present_in_mst[index]) {
        min_weight[tid] = INF; // Set to maximum value to avoid selection
    }
    __syncthreads();

    // Reduction within the block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (min_weight[tid + s] < min_weight[tid]) {
                min_weight[tid] = min_weight[tid + s];
                closest_node[tid] = closest_node[tid + s];
            }
        }
        __syncthreads();
    }

    // Store the minimum value and its index in global memory
    if (tid == 0) {
        d_min_weights[blockIdx.x] = min_weight[0];
        d_min_nodes[blockIdx.x] = closest_node[0];
    }
}

__global__ void update_distances(
        const int *d_matrix, int *d_mst, int *d_distance_vector, int final_min_node, bool *d_present_in_mst) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NODES && tid != final_min_node && !d_present_in_mst[tid]) {
        int index = final_min_node * NODES + tid; // Assuming d_matrix is a flattened 2D array
        if (d_matrix[index] < d_distance_vector[tid]) {
            d_mst[tid] = final_min_node;
            d_distance_vector[tid] = d_matrix[index];
            d_present_in_mst[final_min_node] = true;
        }
    }
}

void parallel_mst(int num_blocks, int nBytes, int *d_matrix, int *d_mst, int *d_distance_vector, int *d_min_weights,
                  int *d_min_nodes, bool *d_present_in_mst) {
    vector<int> distance_vector(NODES);
    for (int i = 0; i < NODES; i++) {
        // Launch kernel with appropriate block and thread configuration
        local_closest_node<<<num_blocks, BLOCK_SIZE, nBytes>>>(
                d_distance_vector, d_min_weights, d_min_nodes, d_present_in_mst);
        cudaDeviceSynchronize();        // Wait for kernel to finish

        int *h_min_weights = new int[num_blocks];
        int *h_min_nodes = new int[num_blocks];

        cudaMemcpy(h_min_weights, d_min_weights, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_min_nodes, d_min_nodes, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);

        int global_min_weight = h_min_weights[0];
        int global_min_node = h_min_nodes[0];

        for (int k = 1; k < num_blocks; ++k) {
            if (h_min_weights[k] < global_min_weight) {
                global_min_weight = h_min_weights[k];
                global_min_node = h_min_nodes[k];
            }
        }

        update_distances<<<num_blocks, BLOCK_SIZE>>>(
                d_matrix, d_mst, d_distance_vector, global_min_node,
                d_present_in_mst);

        cudaDeviceSynchronize();
    }
}

int main() {
    Graph graph = generate(NODES);
    int edges = tot_edges(graph, NODES);
    printf("Generated graph of %d vertices and %d edges:\n", NODES, edges);

    //=====SEQUENTIAL PRIM MST=====
    using clock = std::chrono::system_clock;
    using ms = std::chrono::duration<double, std::milli>;
    auto before = clock::now();

    vector<int> sequential_MST = sequential_prim_MST(graph, NODES);

    ms duration = clock::now() - before;
    printf("Sequential execution time: %f milliseconds\n", duration.count());

    //=====PARALLEL PRIM MST=====
    //HOST MEMORY
    int num_blocks = (NODES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int source = 0;

    vector<int> distance_vector(NODES, INF);
    distance_vector[source] = 0;

    vector<int> mst(NODES, -1);

    int *adj_matrix = new int[NODES * NODES];
    adjacency_list_to_matrix(graph, adj_matrix, NODES);

    //DEVICE MEMORY
    int* d_matrix;      //partitioned adjacency matrix
    cudaMalloc((void **)&d_matrix, NODES * NODES * sizeof(int));
    cudaMemcpy(d_matrix, adj_matrix, NODES * NODES * sizeof(int), cudaMemcpyHostToDevice);

    int* d_distance_vector; //vector distance: distance from the MST to each node
    cudaMalloc((void **)&d_distance_vector, NODES * sizeof(int));
    cudaMemcpy(d_distance_vector, distance_vector.data(), NODES * sizeof(int), cudaMemcpyHostToDevice);

    bool *d_present_in_mst;
    cudaMalloc((void **)&d_present_in_mst, BLOCK_SIZE * sizeof (bool));

    int *d_min_weights;
    cudaMalloc((void **)&d_min_weights, BLOCK_SIZE * sizeof (int));

    int* d_min_nodes;
    cudaMalloc((void **)&d_min_nodes, BLOCK_SIZE * sizeof (int));

    int *d_mst;
    cudaMalloc((void **)&d_mst, NODES * sizeof(int));

    int nBytes = NODES * sizeof(int) + NODES * sizeof(int);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    parallel_mst(num_blocks, nBytes, d_matrix, d_mst, d_distance_vector,
                 d_min_weights, d_min_nodes, d_present_in_mst);
    cudaEventRecord(stop);

    cudaMemcpy(mst.data(), d_mst, NODES * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
        printf("\nUsing %d blocks with size %d\n", num_blocks, BLOCK_SIZE);
    printf("cuda event parallel execution time: %f milliseconds\n", milliseconds);

    // Free memory
    delete[] adj_matrix;
    cudaFree(d_matrix);
    cudaFree(d_distance_vector);
    cudaFree(d_min_weights);
    cudaFree(d_min_nodes);
    cudaFree(d_mst);
    return 0;
}