#include <iostream>
#include <random>
#include <vector>
#include <set>
#include <limits>
#define INF LLONG_MAX
#define MIN_EDGE_WEIGHT 10
#define MAX_EDGE_WEIGHT 100
#define MAX_NODES 100000000
#define BLOCK_SIZE 2
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
__global__ void findClosestNodeLocally(const long long int *matrix, int nodes, long long int *d_localMinNodes,
                                       const long long int *d_distanceVector, const bool *d_presentInMST) {
    __shared__ long long int minWeight;
    __shared__ long long int smallestNodeIndex;

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int elementsPerBlock = nodes / gridDim.x;

    int startIndex = bid * elementsPerBlock;
    int endIndex = (bid + 1) * elementsPerBlock;

    if (tid == 0) {
        minWeight = INF;
    }

    __syncthreads();

    for (int i = startIndex + tid; i < endIndex; i += blockDim.x) {
        // Access and process array elements here
        printf("Block %d, Thread %d: Element %lld present %d\n", bid, tid, d_distanceVector[i], d_presentInMST[i]);
        if(!d_presentInMST[i] && d_distanceVector[i] < minWeight) {
            atomicMin(&minWeight, d_distanceVector[i]);
            atomicExch(reinterpret_cast<unsigned long long int *>(&smallestNodeIndex), i);
        }
    }
    __syncthreads();

    if (tid == 0) {
        d_localMinNodes[bid] = smallestNodeIndex;
        printf("CUDA - from block %d, smallest weight %lld, smallest node %lld\n", bid, minWeight, d_localMinNodes[bid]);
    }

    for(int i = 0 ; i < BLOCK_SIZE ; i++) {
        printf("CUDA - local min nodes: %lld\n", d_localMinNodes[i]);
    }

 /*
     int col = blockIdx.x * blockDim.x + threadIdx.x;
     long long minWeight = INF;
     // Initialize minVal to maximum possible long long value

     */
    /*if (threadIdx.x == 0) {
        minWeight = INF;
    }*/

   /* long long smallestNodeIndex = -1;

    if (col < nodes) {
        for (int row = 0; row < nodes; ++row) {
            // Access matrix[row][col]
            long long index = row * nodes + col;
            //long long element = matrix[index];
            long long element = d_distanceVector[index];
            long long relativeIndex = (row * nodes + col) / nodes;
            printf("index %lld element %lld from block %d\n", index, element, blockIdx.x);
            if(!d_presentInMST[relativeIndex] && element != 0 && element < minWeight) {
                minWeight = element;
                smallestNodeIndex = relativeIndex;
            }
        }
    }

    __syncthreads();
    d_localMinNodes[threadIdx.x] = smallestNodeIndex;*/

    //printf("Closest node %lld with weight %lld from block %d\n", smallestNodeIndex, minWeight, blockIdx.x);
    //d_distanceVector[threadIdx.x] = minWeight;
    //printf("d_result %lld thread %d\n", d_result[threadIdx.x], threadIdx.x);


    /*
    if (threadIdx.x == 0) {
        d_result[threadIdx.x] = minWeight;
    }*/

}

void printDistanceVector(const char *s, vector<long long int> distanceVector) {
    printf("%s", (const char *const) s);
    for(int i = 0 ; i < distanceVector.size(); i++){
        if(i == distanceVector.size() - 1){
            printf("%lld", distanceVector[i]);
        } else {
            printf("%lld, ", distanceVector[i]);
        }
    }
    printf("\n\n");
}

//TODO: unire findClosestNodeLocally findMinIndex ?

__global__ void findMinIndex(
        const long long int *d_distanceVector, int size, int *d_minIndex/*, bool *d_presentInMST, vector<int> d_mst*/) {
    __shared__ int s_minIndex[BLOCK_SIZE];
    __shared__ long long s_minValue[BLOCK_SIZE];

    int localMinIndex = -1;
    long long localMinValue = LLONG_MAX;

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

int main() {
    int nodes = 4;
    int source = 0;
    Graph graph = generate(nodes);
    int edges = totEdges(nodes, graph);

    vector<long long int> distanceVector(nodes, LLONG_MAX); // Key values used to pick minimum weight edge in cut
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

    auto* adjMatrix = new long long[nodes * nodes];
    adjacencyListToMatrix(graph, adjMatrix, nodes);
    printAdjMatrix(adjMatrix, nodes);

    // Allocate memory on device
    int numBlocks = (nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
    long long* d_matrix; //partitioned adjacency matrix
    size_t matrixSize = nodes * nodes * sizeof(long long);
    cudaMalloc((void **)&d_matrix, matrixSize);

    long long* d_distanceVector; //vector distance: distance from the MST to each node
    size_t nodeSize = nodes * sizeof(long long);
    cudaMalloc((void **)&d_distanceVector, nodeSize);
    cudaMemcpy(d_distanceVector, distanceVector.data(), nodeSize, cudaMemcpyHostToDevice);

    bool* d_presentInMST;
    cudaMalloc((void **)&d_presentInMST, nodeSize);

    long long* d_localMinNodes;
    cudaMalloc((void **)&d_localMinNodes, BLOCK_SIZE * sizeof (long long));

    int *d_minIndex;
    cudaMalloc((void**)&d_minIndex, sizeof(int));

    vector<int> d_mst(nodes);
    cudaMalloc((void **)&d_mst, nodeSize);

    // Copy data from host to device
    cudaMemcpy(d_matrix, adjMatrix, matrixSize, cudaMemcpyHostToDevice);

    for (int i = 0; i < nodes - 1; i++) {
        printf("\n===== STEP NUMBER %d ======\n", i+1);
        printDistanceVector("distance vector pre update: ", distanceVector);

        // Launch kernel with appropriate block and thread configuration
        findClosestNodeLocally<<<numBlocks, BLOCK_SIZE>>>(
                d_matrix, nodes, d_localMinNodes, d_distanceVector, d_presentInMST);

        // Wait for kernel to finish
        cudaDeviceSynchronize();

        // Copy result from device to host
        vector<long long> localMinNodes(BLOCK_SIZE);
        cudaMemcpy(localMinNodes.data(), d_localMinNodes, BLOCK_SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(distanceVector.data(), d_distanceVector, nodeSize, cudaMemcpyDeviceToHost);


        for (long long & localMinNode : localMinNodes) {
            cout << "local min nodes: " << localMinNode << endl;
        }



        for(int k = 0 ; k < BLOCK_SIZE ; k++) {
            //if(!presentInMST[k]){
                printf("From block %d, the closest node is %lld with weight %lld\n",
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
        cudaMemcpy(d_presentInMST, presentInMST, nodes * sizeof(long long), cudaMemcpyHostToDevice);

        printf("Index of minimum value: %d\n", minIndex);
        for(int j = 0 ; j < nodes ; j++) {
            if(presentInMST[j]){
                printf("Node %d is present in MST\n", j);
            }
        }

        //TODO: parallel version
        for (const Edge& neighbor : graph[minIndex]) {
            long long int v = neighbor.first;
            long long int weight = neighbor.second;
            if (!presentInMST[v] && weight < distanceVector[v]) {
                mst[v] = minIndex;
                distanceVector[v] = weight;
            }
        }
        cudaMemcpy(d_distanceVector, distanceVector.data(), nodes * sizeof(long long), cudaMemcpyHostToDevice);

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