#include <vector>
#include <utility>
#include <climits>
#include "random_graph_generator.h"
using namespace std;
typedef pair<int, int> Edge; // Define edge type
typedef vector<vector<Edge>> Graph; // Define graph type

// Function to find the minimum weight vertex from the set of vertices not yet included in MST
int min_weight_vertex(int n, vector<int>& key, vector<bool>& not_in_mst) {
    int min_weight = INT_MAX;
    int min_weight_vertex;
    for (int v = 0; v < n; v++) {
        if (not_in_mst[v] && key[v] < min_weight) {
            min_weight = key[v];
            min_weight_vertex = v;
        }
    }
    return min_weight_vertex;
}

// Function to implement Prim's MST algorithm and return MST as a graph
vector<int> sequential_prim_MST(const Graph& graph, int n) {
    vector<int> mst(n, -1); // Array to store constructed MST parent
    vector<int> key(n, INT_MAX); // Key values used to pick minimum weight edge in cut
    vector<bool> not_in_mst(n, true); // To represent set of vertices not yet included in MST

    key[0] = 0; // Make key 0 so that this vertex is picked as first vertex
    mst[0] = -1; // First node is always root of MST

    // The MST will have n vertices
    for (int count = 0; count < n - 1; count++) {
        // Pick the minimum key vertex from the set of vertices not yet included in MST
        int u = min_weight_vertex(n, key, not_in_mst);

        // Add the picked vertex to the MST set
        not_in_mst[u] = false;

        // Update key value and mst index of the adjacent vertices of the picked vertex.
        // Consider only those vertices which are not yet included in MST
        for (const Edge& e : graph[u]) {
            int v = e.first;
            int weight = e.second;
            if (not_in_mst[v] && weight < key[v]) {
                mst[v] = u;
                key[v] = weight;
            }
        }
    }
    return mst;
}