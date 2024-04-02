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
using namespace std;
typedef pair<int, int> Edge; // Define edge type
typedef vector<vector<Edge>> Graph; // Define graph type

void printGraph(int n, const Graph& graph) {
    for (int u = 0; u < n; ++u) {
        for (const auto& neighbor : graph[u]) {
            int v = neighbor.first;
            int weight = neighbor.second;
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
    // Define the distribution for integers
    uniform_int_distribution<int> random_weight(MIN_EDGE_WEIGHT, MAX_EDGE_WEIGHT);
    uniform_int_distribution<int> random_extra_edges(1, ((nodes - 1) * nodes)/2 - (nodes-1));
    uniform_int_distribution<int> random_node(0, nodes-1);

    Graph adjacency(nodes);

    int extra_edges = random_extra_edges(mt);

    if(nodes - 1 + extra_edges > MAX_NODES){
        int difference = MAX_NODES - (nodes - 1);
        uniform_int_distribution<int> random_difference(0, difference +1);
        extra_edges = random_difference(mt);
    }

    vector<int> graph(nodes);

    for(int i = 0; i < nodes; ++i){
        graph[i] = i;
    }

    shuffle(graph.begin(),graph.end(), mt19937(random_device()()));

    set<Edge> present_edge;

    for(int i = 1; i < nodes; ++i){
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

void printAdjMatrix(const int *adjMatrix, int nodes) {
    for (size_t i = 0; i < nodes; ++i) {
        for (size_t j = 0; j < nodes; ++j) {
            int weight = adjMatrix[i * nodes + j];
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

//TODO: unire findClosestNodeLocally findMinIndex ?

__global__ void findMinIndex(
        const int *d_distanceVector, int size, int *d_minIndex/*, bool *d_presentInMST, vector<int> d_mst*/) {
    __shared__ int s_minIndex[BLOCK_SIZE];
    __shared__ int s_minValue[BLOCK_SIZE];

    int localMinIndex = -1;
    int localMinValue = INF;

    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        if (d_distanceVector[i] < localMinValue) {
            localMinValue = d_distanceVector[i];
            localMinIndex = i;
        }
    }

    s_minValue[threadIdx.x] = localMinValue;
    s_minIndex[threadIdx.x] = localMinIndex;

    __syncthreads();

    // Perform block-level reduction to find global minimum value and its index
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (s_minValue[threadIdx.x + stride] < s_minValue[threadIdx.x]) {
                s_minValue[threadIdx.x] = s_minValue[threadIdx.x + stride];
                s_minIndex[threadIdx.x] = s_minIndex[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    // Write global minimum value index to output array
    if (threadIdx.x == 0) {
        *d_minIndex = s_minIndex[0];
        //d_presentInMST[s_minIndex[0]] = true;
    }
}

__global__ void findClosestNodeLocally(int nodes, int *d_localMinNodes, const int *d_distanceVector,
                                       const bool *d_presentInMST) {
    __shared__ int minWeight;
    __shared__ int smallestNodeIndex;

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int elementsPerBlock = nodes / gridDim.x;

    int startIndex = bid * elementsPerBlock;
    int endIndex = (bid + 1) * elementsPerBlock;

    if (tid == 0) {
        minWeight = INF;
        smallestNodeIndex = startIndex + tid;
    }

    __syncthreads();

    for (int i = startIndex + tid; i < endIndex; i += blockDim.x) {
        // Access and process array elements here
        printf("Block %d, Thread %d: Element %d present %d\n", bid, tid, d_distanceVector[i], d_presentInMST[i]);
        if(!d_presentInMST[i] && d_distanceVector[i] < minWeight) {
            atomicMin(&minWeight, d_distanceVector[i]);
            atomicExch(&smallestNodeIndex, i);
        }
    }
    __syncthreads();

    if (tid == 0) {
        d_localMinNodes[bid] = smallestNodeIndex;
        printf("CUDA - from block %d, smallest weight %d, smallest node %d\n", bid, minWeight, d_localMinNodes[bid]);
    }
}

int main() {
    int nodes = 4;
    int source = 0;
    Graph graph = generate(nodes);
    int edges = totEdges(nodes, graph);

    vector<int> distanceVector(nodes, INF); // Key values used to pick minimum weight edge in cut
    distanceVector[0] = 0;
    vector<int> mst(nodes); // Array to store constructed MST parent, MST has numVertices-1 edges
    auto *presentInMST = new bool[nodes]; // To represent set of vertices not yet included in MST
    for (int i = 0; i < nodes; i++) {
        if(i == source) {
            presentInMST[i] = true;
        }
        presentInMST[i] = false;
    }

    printf("Generated graph of %d vertices and %d edges:\n", nodes, edges);
    printGraph(nodes, graph);

    auto* adjMatrix = new int[nodes * nodes];
    adjacencyListToMatrix(graph, adjMatrix, nodes);
    printAdjMatrix(adjMatrix, nodes);

    // Allocate memory on device
    int numBlocks = (nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int* d_matrix; //partitioned adjacency matrix
    size_t matrixSize = nodes * nodes * sizeof(int);
    cudaMalloc((void **)&d_matrix, matrixSize);

    int* d_distanceVector; //vector distance: distance from the MST to each node
    size_t nodeSize = nodes * sizeof(int);
    cudaMalloc((void **)&d_distanceVector, nodeSize);
    cudaMemcpy(d_distanceVector, distanceVector.data(), nodeSize, cudaMemcpyHostToDevice);

    bool* d_presentInMST;
    cudaMalloc((void **)&d_presentInMST, nodeSize);

    int* d_localMinNodes;
    cudaMalloc((void **)&d_localMinNodes, BLOCK_SIZE * sizeof (int));

    int *d_minIndex;
    cudaMalloc((void**)&d_minIndex, sizeof(int));

    vector<int> d_mst(nodes);
    cudaMalloc((void **)&d_mst, nodeSize);

    // Copy data from host to device
    cudaMemcpy(d_matrix, adjMatrix, matrixSize, cudaMemcpyHostToDevice);

    for (int i = 0; i < nodes - 1; i++) {
        printf("\n===== STEP NUMBER %d ======\n", i+1);
        printDistanceVector("distance vector pre update: ", distanceVector);

        vector<int> localMinNodes(BLOCK_SIZE, 0);
        cudaMemcpy(d_localMinNodes, localMinNodes.data(), BLOCK_SIZE * sizeof(int), cudaMemcpyHostToDevice);

        // Launch kernel with appropriate block and thread configuration
        findClosestNodeLocally<<<numBlocks, BLOCK_SIZE>>>(nodes, d_localMinNodes, d_distanceVector, d_presentInMST);

        // Wait for kernel to finish
        cudaDeviceSynchronize();

        // Copy result from device to host
        cudaMemcpy(localMinNodes.data(), d_localMinNodes, BLOCK_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(distanceVector.data(), d_distanceVector, nodeSize, cudaMemcpyDeviceToHost);

        for(int k = 0 ; k < BLOCK_SIZE ; k++) {
            //if(!presentInMST[k]){
                printf("From block %d, the closest node is %d with weight %d\n",
                       k, localMinNodes[k], distanceVector[localMinNodes[k]]);
            //}
        }
        printf("\n\n");

        // Launch kernel to find index of minimum value
        findMinIndex<<<numBlocks, BLOCK_SIZE>>>(
                d_distanceVector, nodes, d_minIndex/*, d_presentInMST, d_mst*/);

        // Copy result back to host
        int minIndex = -1;
        cudaMemcpy(&minIndex, d_minIndex, sizeof(int), cudaMemcpyDeviceToHost);
        presentInMST[minIndex] = true;
        cudaMemcpy(d_presentInMST, presentInMST, nodes * sizeof(int), cudaMemcpyHostToDevice);

        printf("Index of minimum value: %d\n", minIndex);
        for(int j = 0 ; j < nodes ; j++) {
            if(presentInMST[j]){
                printf("Node %d is present in MST\n", j);
            }
        }

        //TODO: parallel version
        for (const Edge& neighbor : graph[minIndex]) {
            int v = neighbor.first;
            int weight = neighbor.second;
            if (!presentInMST[v] && weight < distanceVector[v]) {
                mst[v] = minIndex;
                distanceVector[v] = weight;
            }
        }
        cudaMemcpy(d_distanceVector, distanceVector.data(), nodes * sizeof(int), cudaMemcpyHostToDevice);

        printDistanceVector("distance vector post update: ", distanceVector);
    }

    // Construct MST graph from mst array
    Graph mstGraph(nodes);
    for (int i = 0 ; i < nodes; ++i) {
        int u = mst[i];
        mstGraph[u].emplace_back(i, distanceVector[i]);
    }
    printGraph(nodes, mstGraph);

    /*
    printf("Index of minimum value: %d\n", minIndex);
    for(int i = 0 ; i < nodes ; i++) {
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