#include <iostream>
#include <random>
#include <vector>
#include <set>
#include <limits>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>
#define INF INT_MAX
#define MIN_EDGE_WEIGHT 10
#define MAX_EDGE_WEIGHT 100
#define MAX_NODES 100000000
#define BLOCK_SIZE 512
#define NODES 2048
using namespace std;
typedef pair<int, int> Edge; // Define edge type
typedef vector<vector<Edge>> Graph; // Define graph type

void print_graph(const Graph& graph) {
    for (int u = 0; u < NODES; ++u) {
        for (const auto& neighbor : graph[u]) {
            int v = neighbor.first;
            int weight = neighbor.second;
            cout << u << " " << v << " " << weight << endl;
            //cout << u << " - " << v << " : " << weight << endl;
        }
    }
    printf("\n");
}

Graph generate() {
    printf("generating graph...\n");
    // Use a random device to seed the random number engine
    random_device rd;
    // Use the Mersenne Twister engine for randomness
    mt19937 mt(rd());
    // Define the distribution for integers
    uniform_int_distribution<int> random_weight(MIN_EDGE_WEIGHT, MAX_EDGE_WEIGHT);
    uniform_int_distribution<int> random_extra_edges(1, ((NODES - 1) * NODES)/2 - (NODES-1));
    uniform_int_distribution<int> random_node(0, NODES-1);

    Graph adjacency(NODES);

    int extra_edges = random_extra_edges(mt);

    if(NODES - 1 + extra_edges > MAX_NODES){
        int difference = MAX_NODES - (NODES - 1);
        uniform_int_distribution<int> random_difference(0, difference +1);
        extra_edges = random_difference(mt);
    }

    vector<int> graph(NODES);

    for(int i = 0; i < NODES; ++i){
        graph[i] = i;
    }

    shuffle(graph.begin(),graph.end(), mt19937(random_device()()));

    set<Edge> present_edge;

    for(int i = 1; i < NODES; ++i){
        uniform_int_distribution<int> random_add(0, i - 1);
        int add = random_add(mt);
        int weight = random_weight(mt);
        adjacency[graph[i]].emplace_back(graph[add], weight);
        adjacency[graph[add]].emplace_back(graph[i], weight);
        present_edge.insert(make_pair(min(graph[add], graph[i]), max(graph[add], graph[i])));
    }

    for(int i = 1; i <= extra_edges; ++i){
        int weight = random_weight(mt);
        while(true){
            int node1 = random_node(mt);
            int node2 = random_node(mt);
            if(node1 == node2) continue;
            if(present_edge.find(make_pair(min(node1, node2), max(node1, node2))) == present_edge.end()){
                adjacency[node1].emplace_back(node2, weight);
                adjacency[node2].emplace_back(node1, weight);
                present_edge.insert(make_pair(min(node1, node2), max(node1, node2)));
                break;
            }
        }
    }
    return adjacency;
}

int tot_edges(int n, const Graph& graph) {
    int count = 0;
    for (int u = 0; u < n; ++u) {
        count += (int) graph[u].size();
    }
    return count;
}

void adjacency_list_to_matrix(const vector<vector<pair<int, int>>>& adjList, int* adj_matrix, size_t n) {
    // Initialize the adjacency matrix with INF initially
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            adj_matrix[i * n + j] = INF;
        }
    }

    // Populate the adjacency matrix with appropriate values from the adjacency list
    for (size_t i = 0; i < n; ++i) {
        for (const auto& edge : adjList[i]) {
            int vertex = edge.first;
            int weight = edge.second;
            adj_matrix[i * n + vertex] = weight;
        }
    }

    // Diagonal elements should be 0 (no self-loops)
    for (size_t i = 0; i < n; ++i) {
        adj_matrix[i * n + i] = 0;
    }
}

void print_adj_matrix(const int *adj_matrix) {
    for (size_t i = 0; i < NODES; ++i) {
        for (size_t j = 0; j < NODES; ++j) {
            int weight = adj_matrix[i * NODES + j];
            if (weight == INF) {
                cout << "INF\t";
            } else {
                cout << weight << "\t";
            }
        }
        cout << endl;
    }
}

//=========================================CUDA===============================================
void print_distance_vector(vector<int> distance_vector) {
    printf("distance vector: ");
    for(int i = 0 ; i < distance_vector.size(); i++){
        if(i == distance_vector.size() - 1){
            printf("%d", distance_vector[i]);
        } else {
            printf("%d, ", distance_vector[i]);
        }
    }
    printf("\n\n");
}

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
        //printf("local - min w %d, min n %d\n", min_weight[0], closest_node[0]);
    }
}

__global__ void update_distances(
        const int *d_matrix, int *d_mst, int *d_distance_vector, int final_min_node, const bool *d_present_in_mst) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NODES && tid != final_min_node && !d_present_in_mst[tid]) {
        int index = final_min_node * NODES + tid; // Assuming d_matrix is a flattened 2D array
        if (d_matrix[index] < d_distance_vector[tid]) {
            d_mst[tid] = final_min_node;
            d_distance_vector[tid] = d_matrix[index];
        }
    }
}


int main() {
    //HOST MEMORY
    int numBlocks = (NODES + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int source = 0;
    Graph graph = generate();
    int edges = tot_edges(NODES, graph);
    printf("Generated graph of %d vertices and %d edges:\n", NODES, edges);
    //print_graph(graph);

    vector<int> distance_vector(NODES, INF);
    distance_vector[source] = 0;

    vector<int> mst(NODES, -1);
    bool present_in_mst[NODES] = { false };

    int *adj_matrix = new int[NODES * NODES];
    adjacency_list_to_matrix(graph, adj_matrix, NODES);
    //print_adj_matrix(adj_matrix);

    //DEVICE MEMORY
    int* d_matrix;      //partitioned adjacency matrix
    cudaMalloc((void **)&d_matrix, NODES * NODES * sizeof(int));
    cudaMemcpy(d_matrix, adj_matrix, NODES * NODES * sizeof(int), cudaMemcpyHostToDevice);

    int* d_distance_vector; //vector distance: distance from the MST to each node
    cudaMalloc((void **)&d_distance_vector, NODES * sizeof(int));
    cudaMemcpy(d_distance_vector, distance_vector.data(), NODES * sizeof(int), cudaMemcpyHostToDevice);

    bool* d_present_in_mst;
    cudaMalloc((void **)&d_present_in_mst, NODES * sizeof(int));

    int* d_min_weights;
    cudaMalloc((void **)&d_min_weights, BLOCK_SIZE * sizeof (int));

    int* d_min_nodes;
    cudaMalloc((void **)&d_min_nodes, BLOCK_SIZE * sizeof (int));

    int *d_mst;
    cudaMalloc((void **)&d_mst, NODES * sizeof(int));

    int nBytes = NODES * sizeof(int) + NODES * sizeof(int);
    for (int i = 0; i < NODES; i++) {
        //printf("\n===== STEP NUMBER %d ======\n", i + 1);
        //print_distance_vector(distance_vector);

        // Launch kernel with appropriate block and thread configuration
        local_closest_node<<<numBlocks, BLOCK_SIZE, nBytes>>>(
                d_distance_vector, d_min_weights, d_min_nodes, d_present_in_mst);
        cudaDeviceSynchronize();        // Wait for kernel to finish

        //Find global closest node from the local solutions
        thrust::device_ptr<int> thrust_weights_ptr(d_min_weights);
        thrust::device_ptr<int> thrust_nodes_ptr = thrust::device_pointer_cast(d_min_nodes);
        thrust::device_ptr<int> min_ptr = thrust::min_element(
                thrust::device, thrust_weights_ptr, thrust_weights_ptr + numBlocks);

        int final_min_weight = *min_ptr;
        int final_min_node = thrust_nodes_ptr[min_ptr - thrust_weights_ptr];
        present_in_mst[final_min_node] = true;
        cudaMemcpy(d_present_in_mst, present_in_mst, NODES * sizeof(bool), cudaMemcpyHostToDevice);

/*
        printf("Minimum weight: %d, node: %d\n", final_min_weight, final_min_node);
        for(int j = 0 ; j < NODES ; j++) {
            if(present_in_mst[j]){
                printf("Node %d is present in MST\n", j);
            }
        }
*/
        update_distances<<<numBlocks, BLOCK_SIZE>>>(
                d_matrix, d_mst, d_distance_vector, final_min_node, d_present_in_mst);

        cudaDeviceSynchronize();

        cudaMemcpy(distance_vector.data(), d_distance_vector, NODES * sizeof(int), cudaMemcpyDeviceToHost);
    }

    cudaMemcpy(mst.data(), d_mst, NODES * sizeof(int), cudaMemcpyDeviceToHost);

    // Construct MST graph from mst array
    Graph mstGraph(NODES);
    for (int i = 1 ; i < NODES; ++i) {
        int u = mst[i];
        mstGraph[u].emplace_back(i, distance_vector[i]);
    }
    printf("\nThe MST is:\n");
    //print_graph(mstGraph);

    // Free memory
    delete[] adj_matrix;
    cudaFree(d_matrix);
    cudaFree(d_distance_vector);
    cudaFree(d_present_in_mst);
    cudaFree(d_min_weights);
    cudaFree(d_min_nodes);
    cudaFree(d_mst);
    return 0;
}