#include <iostream>
#include <random>
#include <vector>
#include <set>
#include <limits>
#define INF LLONG_MAX
#define MIN_EDGE_WEIGHT 10
#define MAX_EDGE_WEIGHT 100
#define MAX_NODES 100000000
#define BLOCK_SIZE 256
using namespace std;
typedef pair<long long int, long long int> Edge; // Define edge type
typedef vector<vector<Edge>> Graph; // Define graph type

void printGraph(int n, const Graph& graph) {
    for (int u = 0; u < n; ++u) {
        for (const auto& neighbor : graph[u]) {
            long long v = neighbor.first;
            long long weight = neighbor.second;
            cout << u << " - " << v << " : " << weight << endl;
        }
    }
    printf("\n");
}

Graph generate(int nodes) {
    // Use a random device to seed the random number engine
    random_device rd;
    // Use the Mersenne Twister engine for randomness
    mt19937 mt(rd());
    // Define the distribution for long long integers
    uniform_int_distribution<long long> random_weight(MIN_EDGE_WEIGHT, MAX_EDGE_WEIGHT);
    uniform_int_distribution<long long> random_extra_edges(1, ((nodes - 1) * nodes)/2 - (nodes-1));
    uniform_int_distribution<long long> random_node(0, nodes-1);

    Graph adjacency(nodes);

    long long extra_edges = random_extra_edges(mt);

    if(nodes - 1 + extra_edges > MAX_NODES){
        long long difference = MAX_NODES - (nodes - 1);
        uniform_int_distribution<long long> random_difference(0, difference +1);
        extra_edges = random_difference(mt);
    }

    vector<long long> graph(nodes);

    for(int i = 0; i < nodes; ++i){
        graph[i] = i;
    }

    shuffle(graph.begin(),graph.end(), mt19937(random_device()()));

    set<Edge> present_edge;

    for(int i = 1; i < nodes; ++i){
        uniform_int_distribution<long long> random_add(0, i - 1);
        long long add = random_add(mt);
        long long weight = random_weight(mt);
        adjacency[graph[i]].emplace_back(graph[add], weight);
        adjacency[graph[add]].emplace_back(graph[i], weight);
        present_edge.insert(make_pair(min(graph[add], graph[i]), max(graph[add], graph[i])));
    }

    for(int i = 1; i <= extra_edges; ++i){
        long long weight = random_weight(mt);
        while(true){
            long long node1 = random_node(mt);
            long long node2 = random_node(mt);
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

int totEdges(int n, const Graph& graph) {
    int count = 0;
    for (int u = 0; u < n; ++u) {
        count += graph[u].size();
    }
    return count;
}

void adjacencyListToMatrix(const vector<vector<pair<long long, long long>>>& adjList, long long* adjMatrix, size_t n) {
    // Initialize the adjacency matrix with INF initially
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            adjMatrix[i * n + j] = INF;
        }
    }

    // Populate the adjacency matrix with appropriate values from the adjacency list
    for (size_t i = 0; i < n; ++i) {
        for (const auto& edge : adjList[i]) {
            long long vertex = edge.first;
            long long weight = edge.second;
            adjMatrix[i * n + vertex] = weight;
        }
    }

    // Diagonal elements should be 0 (no self-loops)
    for (size_t i = 0; i < n; ++i) {
        adjMatrix[i * n + i] = 0;
    }
}

void printAdjMatrix(const long long *adjMatrix, int nodes) {
    for (size_t i = 0; i < nodes; ++i) {
        for (size_t j = 0; j < nodes; ++j) {
            long long weight = adjMatrix[i * nodes + j];
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
__global__ void findClosestNodeLocally(const long long int *matrix, int nodes, long long int *d_result) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    long long minWeight = INF;
    // Initialize minVal to maximum possible long long value
    /*if (threadIdx.x == 0) {
        minWeight = INF;
    }*/
    long long smallestNodeIndex = 0;


    if (col < nodes) {
        for (int row = 0; row < nodes; ++row) {
            // Access matrix[row][col]
            long long index = row * nodes + col;
            long long element = matrix[index];
            if(element != 0 && element < minWeight) {
                minWeight = element;
                smallestNodeIndex = (row * nodes + col) / nodes;
            }
        }
    }

    __syncthreads();

    printf("Smallest node %lld with weight %lld from thread %d\n", smallestNodeIndex, minWeight, threadIdx.x);
    d_result[threadIdx.x] = minWeight;
    //printf("d_result %lld thread %d\n", d_result[threadIdx.x], threadIdx.x);


    /*
    if (threadIdx.x == 0) {
        d_result[threadIdx.x] = minWeight;
    }*/

}


int main() {
    int nodes = 4;
    Graph graph = generate(nodes);
    int edges = totEdges(nodes, graph);

    printf("Generated graph of %d vertices and %d edges:\n", nodes, edges);
    printGraph(nodes, graph);

    auto* adjMatrix = new long long[nodes * nodes];
    adjacencyListToMatrix(graph, adjMatrix, nodes);
    printAdjMatrix(adjMatrix, nodes);

    // Allocate memory on device
    int numBlocks = (nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
    long long* d_matrix;
    size_t matrixSize = nodes * nodes * sizeof(long long);
    cudaMalloc((void **)&d_matrix, matrixSize);
    long long* d_result;
    size_t resultSize = nodes * sizeof(long long);
    cudaMalloc((void **)&d_result, resultSize);

    // Copy data from host to device
    cudaMemcpy(d_matrix, adjMatrix, matrixSize, cudaMemcpyHostToDevice);

    // Launch kernel with appropriate block and thread configuration
    findClosestNodeLocally<<<numBlocks, nodes>>>(d_matrix, nodes, d_result);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy result from device to host
    vector<long long> minValues(nodes);
    cudaMemcpy(minValues.data(), d_result, resultSize, cudaMemcpyDeviceToHost);
    for(int i = 0 ; i < nodes ; i++) {
        printf("Minimum value %lld from node %d \n", minValues[i], i);
    }

    // Free memory
    free(adjMatrix);
    cudaFree(d_result);
    cudaFree(d_matrix);

    return 0;
}