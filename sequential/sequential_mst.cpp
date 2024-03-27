#include <iostream>
#include <vector>
#include <utility>
#include <climits>
#include "random_graph_generator.h"

using namespace std;

typedef pair<long long int, long long int> Edge; // Define edge type
typedef vector<vector<Edge>> Graph; // Define graph type

// Function to find the minimum weight vertex from the set of vertices not yet included in MST
int minWeightVertex(int n, vector<long long int>& key, vector<bool>& notInMst) {
    long long int minWeight = LLONG_MAX;
    int minWeightVertex;
    for (int v = 0; v < n; v++) {
        if (notInMst[v] && key[v] < minWeight) {
            minWeight = key[v];
            minWeightVertex = v;
        }
    }
    return minWeightVertex;
}

// Function to implement Prim's MST algorithm and return MST as a graph
Graph primMST(int n, const Graph& graph) {
    vector<long long int> key(n, LLONG_MAX); // Key values used to pick minimum weight edge in cut
    vector<bool> notInMst(n, true); // To represent set of vertices not yet included in MST
    vector<int> mst(n, -1); // Array to store constructed MST parent

    key[0] = 0; // Make key 0 so that this vertex is picked as first vertex
    mst[0] = -1; // First node is always root of MST


    // The MST will have n vertices
    for (int count = 0; count < n - 1; count++) {
        // Pick the minimum key vertex from the set of vertices not yet included in MST

        int u = minWeightVertex(n, key, notInMst);


        // Add the picked vertex to the MST set
        notInMst[u] = false;

        // Update key value and mst index of the adjacent vertices of the picked vertex.
        // Consider only those vertices which are not yet included in MST

        for (const Edge& neighbor : graph[u]) {
            long long int v = neighbor.first;
            long long int weight = neighbor.second;
            if (notInMst[v] && weight < key[v]) {
                mst[v] = u;
                key[v] = weight;
            }
        }
    }


    // Construct MST graph from mst array
    Graph mstGraph(n);
    for (int i = 1; i < n; ++i) {
        int u = mst[i];
        mstGraph[u].emplace_back(i, key[i]);
    }

    return mstGraph;
}

void printGraph(int n, const Graph& graph) {
    for (int u = 0; u < n; ++u) {
        for (const auto& neighbor : graph[u]) {
            long long int v = neighbor.first;
            long long int weight = neighbor.second;
            cout << u << " - " << v << " : " << weight << endl;
        }
    }
    printf("\n");
}

int main() {
    int n = 10;
    Graph graph = generate(n);

    printf("Generated graph:\n");
    printGraph(n, graph);

    // Get MST
    Graph mst = primMST(n, graph);

    // Printing MST (for verification)
    printf("Edges of MST (node - node : weight):\n");
    printGraph(n, mst);

    return 0;
}