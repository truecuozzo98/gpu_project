#include <iostream>
#include <random>
#include <vector>
#include <set>
#include <limits>
#define INF LLONG_MAX
#define MAX_EDGE_WEIGHT 100
#define MAX_NODES 100000000
#define MIN_EDGE_WEIGHT 10
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
    Graph adjacency(nodes);

    long long extra_edges = ((nodes - 1) * nodes)/2 - (nodes-1);
    extra_edges = rand() % (extra_edges + 1);

    if(nodes - 1 + extra_edges > MAX_NODES){
        long long difference = MAX_NODES - (nodes - 1);
        extra_edges = rand() % (difference + 1);
    }

    vector<long long> graph(nodes);

    for(int i = 0; i < nodes; ++i){
        graph[i] = i;
    }

    shuffle(graph.begin(),graph.end(), std::mt19937(std::random_device()()));

    set<Edge> present_edge;

    for(int i = 1; i < nodes; ++i){
        long long add = rand() % i;
        long long weight = rand() % MAX_EDGE_WEIGHT;
        adjacency[graph[i]].emplace_back(graph[add], weight);
        adjacency[graph[add]].emplace_back(graph[i], weight);
        present_edge.insert(make_pair(min(graph[add], graph[i]), max(graph[add], graph[i])));
    }

    for(int i = 1; i <= extra_edges; ++i){
        long long weight = rand() % MAX_EDGE_WEIGHT;
        while(true){
            long long node1 = rand() % nodes;
            long long node2 = rand() % nodes;
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
__global__ void findClosestNodeLocally(const long long* matrix, int nodes) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    long long minWeight = INF;
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

    printf("Smallest node %lld with weight %lld from thread %d\n", smallestNodeIndex, minWeight, threadIdx.x);
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
    long long* d_matrix;
    size_t matrixSize = nodes * nodes * sizeof(long long);
    cudaMalloc((void **)&d_matrix, matrixSize);

    // Copy data from host to device
    cudaMemcpy(d_matrix, adjMatrix, matrixSize, cudaMemcpyHostToDevice);

    // Launch kernel with appropriate block and thread configuration
    int numBlocks = (nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
    findClosestNodeLocally<<<numBlocks, nodes>>>(d_matrix, nodes);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Free memory
    free(adjMatrix);
    cudaFree(d_matrix);

    return 0;
}