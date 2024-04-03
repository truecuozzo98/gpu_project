#include <iostream>
#include <random>
#include <vector>
#include <set>
#include <limits>
#define INF INT_MAX
#define MIN_EDGE_WEIGHT 10
#define MAX_EDGE_WEIGHT 100
#define MAX_NODES 100000000
#define BLOCK_SIZE 2
#define NODES 4
using namespace std;
typedef pair<int, int> Edge; // Define edge type
typedef vector<vector<Edge>> Graph; // Define graph type

void printGraph(const Graph& graph) {
    for (int u = 0; u < NODES; ++u) {
        for (const auto& neighbor : graph[u]) {
            int v = neighbor.first;
            int weight = neighbor.second;
            cout << u << " - " << v << " : " << weight << endl;
        }
    }
    printf("\n");
}

Graph generate() {
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

int totEdges(int n, const Graph& graph) {
    int count = 0;
    for (int u = 0; u < n; ++u) {
        count += graph[u].size();
    }
    return count;
}

void adjacencyListToMatrix(const vector<vector<pair<int, int>>>& adjList, int* adjMatrix, size_t n) {
    // Initialize the adjacency matrix with INF initially
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            adjMatrix[i * n + j] = INF;
        }
    }

    // Populate the adjacency matrix with appropriate values from the adjacency list
    for (size_t i = 0; i < n; ++i) {
        for (const auto& edge : adjList[i]) {
            int vertex = edge.first;
            int weight = edge.second;
            adjMatrix[i * n + vertex] = weight;
        }
    }

    // Diagonal elements should be 0 (no self-loops)
    for (size_t i = 0; i < n; ++i) {
        adjMatrix[i * n + i] = 0;
    }
}

void printAdjMatrix(const int *adjMatrix) {
    for (size_t i = 0; i < NODES; ++i) {
        for (size_t j = 0; j < NODES; ++j) {
            int weight = adjMatrix[i * NODES + j];
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
void printDistanceVector(const char *s, vector<int> distanceVector) {
    printf("%s", (const char *const) s);
    for(int i = 0 ; i < distanceVector.size(); i++){
        if(i == distanceVector.size() - 1){
            printf("%d", distanceVector[i]);
        } else {
            printf("%d, ", distanceVector[i]);
        }
    }
    printf("\n\n");
}


__global__ void localClosestNode(const int *d_distanceVector, int *d_minWeights, int *d_minNodes,
                                 const bool *d_presentInMST) {
    __shared__ int minWeight[NODES];
    __shared__ int smallestNodeIndex[NODES];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    minWeight[tid] = d_distanceVector[index];
    smallestNodeIndex[tid] = index;
    __syncthreads();

    /*for (int i = 0 ; i<NODES; i++) {
        printf("node: %d present: %d\n", i, d_presentInMST[i]);
    }*/

    // Ignore the elements that are already present in MST
    if (d_presentInMST[index]) {
        minWeight[tid] = INF; // Set to maximum value to avoid selection
    }
    __syncthreads();

    // Reduction within the block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            printf("block %d thread %d tid+s %d index %d\n", bid, tid, tid+s, index);
            if (minWeight[tid + s] < minWeight[tid]) {
                minWeight[tid] = minWeight[tid + s];
                smallestNodeIndex[tid] = smallestNodeIndex[tid + s];
            }
        }
        __syncthreads();
    }

    // Store the minimum value and its index in global memory
    if (tid == 0) {
        d_minWeights[blockIdx.x] = minWeight[0];
        d_minNodes[blockIdx.x] = smallestNodeIndex[0];
        printf("local - block: %d, min_node: %d, min_weight: %d\n", bid, smallestNodeIndex[0], minWeight[0]);
    }
}


__global__ void globalClosestNode(int *min_val, int *min_index, bool *d_presentInMST/*, vector<int> d_mst*/) {
    __shared__ int s_minIndex[NODES];
    __shared__ int s_minValue[NODES];

    int tid = threadIdx.x;

    // Load data into shared memory
    s_minIndex[tid] = min_index[tid];
    s_minValue[tid] = min_val[tid];

    __syncthreads();

    for (int s = NODES / 2; s > 0; s /= 2) {
        if (tid < s) {
            if (s_minValue[tid + s] < s_minValue[tid] && s_minValue[tid+s] != 0) {
                s_minValue[tid] = s_minValue[tid + s];
                s_minIndex[tid] = s_minIndex[tid + s /*+ (threadIdx.x * blockDim.x)*/];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        min_val[0] = s_minValue[0];
        min_index[0] = s_minIndex[0];
        //d_presentInMST[s_minValue[0]] = true;
    }
}

int main() {
    int source = 0;
    Graph graph = generate();
    int edges = totEdges(NODES, graph);

    vector<int> distanceVector(NODES, INF); // Key values used to pick minimum weight edge in cut
    distanceVector[source] = 0;
    vector<int> mst(NODES); // Array to store constructed MST parent, MST has numVertices-1 edges
    auto *presentInMST = new bool[NODES]; // To represent set of vertices not yet included in MST
    for (int i = 0; i < NODES; i++) {
        if(i == source) {
            presentInMST[i] = true;
        }
        presentInMST[i] = false;
    }

    printf("Generated graph of %d vertices and %d edges:\n", NODES, edges);
    printGraph(graph);

    auto* adjMatrix = new int[NODES * NODES];
    adjacencyListToMatrix(graph, adjMatrix, NODES);
    printAdjMatrix(adjMatrix);

    // Allocate memory on device
    int numBlocks = (NODES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int* d_matrix; //partitioned adjacency matrix
    size_t matrixSize = NODES * NODES * sizeof(int);
    cudaMalloc((void **)&d_matrix, matrixSize);

    int* d_distanceVector; //vector distance: distance from the MST to each node
    size_t nodeSize = NODES * sizeof(int);
    cudaMalloc((void **)&d_distanceVector, nodeSize);
    cudaMemcpy(d_distanceVector, distanceVector.data(), nodeSize, cudaMemcpyHostToDevice);

    bool* d_presentInMST;
    cudaMalloc((void **)&d_presentInMST, nodeSize);

    int* d_minWeights;
    cudaMalloc((void **)&d_minWeights, BLOCK_SIZE * sizeof (int));

    int* d_minNodes;
    cudaMalloc((void **)&d_minNodes, BLOCK_SIZE * sizeof (int));

    int *d_minIndex;
    cudaMalloc((void**)&d_minIndex, sizeof(int));

    vector<int> d_mst(NODES);
    cudaMalloc((void **)&d_mst, nodeSize);

    // Copy data from host to device
    cudaMemcpy(d_matrix, adjMatrix, matrixSize, cudaMemcpyHostToDevice);

    for (int i = 0; i < NODES - 1; i++) {
        printf("\n===== STEP NUMBER %d ======\n", i+1);
        printDistanceVector("distance vector pre update: ", distanceVector);

        // Launch kernel with appropriate block and thread configuration
        localClosestNode<<<numBlocks, BLOCK_SIZE>>>(d_distanceVector, d_minWeights, d_minNodes, d_presentInMST);
        // Wait for kernel to finish
        cudaDeviceSynchronize();

        globalClosestNode<<<1, BLOCK_SIZE>>>(d_minWeights, d_minNodes, d_presentInMST);
        cudaDeviceSynchronize();

        int final_min_val, final_min_index;
        cudaMemcpy(&final_min_val, d_minWeights, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&final_min_index, d_minNodes, sizeof(int), cudaMemcpyDeviceToHost);
        //cudaMemcpy(presentInMST, d_presentInMST, NODES * sizeof(int), cudaMemcpyDeviceToHost);

        printf("Minimum value: %d, Index: %d\n", final_min_val, final_min_index);
        presentInMST[final_min_index] = true;
        for(int j = 0 ; j < NODES ; j++) {
            if(presentInMST[j]){
                printf("Node %d is present in MST\n", j);
            }
        }

        cudaMemcpy(d_presentInMST, presentInMST, NODES * sizeof(int), cudaMemcpyHostToDevice);

        /*
        // Launch kernel to find index of minimum value
        globalClosestNode<<<numBlocks, BLOCK_SIZE>>>(
                d_distanceVector, NODES, d_minIndex, d_presentInMST, d_mst);
        cudaDeviceSynchronize();

        int final_min_val, final_min_index;
        cudaMemcpy(&final_min_val, d_minWeights, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&final_min_index, d_minNodes, sizeof(int), cudaMemcpyDeviceToHost);
        */

        /*
        printf("Minimum value: %d, Index: %d\n", final_min_val, final_min_index);

        // Copy result back to host
        int minIndex = -1;
        cudaMemcpy(&minIndex, d_minIndex, sizeof(int), cudaMemcpyDeviceToHost);
        presentInMST[minIndex] = true;
        cudaMemcpy(d_presentInMST, presentInMST, NODES * sizeof(int), cudaMemcpyHostToDevice);

        printf("Index of minimum value: %d\n", minIndex);
        for(int j = 0 ; j < NODES ; j++) {
            if(presentInMST[j]){
                printf("Node %d is present in MST\n", j);
            }
        }*/

        //TODO: parallel version
        for (const Edge& neighbor : graph[final_min_index]) {
            int v = neighbor.first;
            int weight = neighbor.second;
            if (!presentInMST[v] && weight < distanceVector[v]) {
                mst[v] = final_min_index;
                distanceVector[v] = weight;
            }
        }
        cudaMemcpy(d_distanceVector, distanceVector.data(), NODES * sizeof(int), cudaMemcpyHostToDevice);

        printDistanceVector("distance vector post update: ", distanceVector);
    }

    // Construct MST graph from mst array
    Graph mstGraph(NODES);
    for (int i = 0 ; i < NODES; ++i) {
        int u = mst[i];
        mstGraph[u].emplace_back(i, distanceVector[i]);
    }
    printGraph(mstGraph);

    /*
    printf("Index of minimum value: %d\n", minIndex);
    for(int i = 0 ; i < NODES ; i++) {
        if(presentInMST[i]){
            printf("Node %d is present in MST\n", i);
        }
    }*/

    // Free memory
    free(adjMatrix);
    free(presentInMST);
    //cudaFree(d_result);
    cudaFree(d_matrix);

    return 0;
}