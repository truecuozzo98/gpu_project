#include <stdio.h>
#include <stdbool.h>
#include <limits.h>
#include <stdlib.h>
#include <vector>
#include <tuple>
#include "random_graph_generator.h"

using namespace std;

// Function to find the vertex with minimum key value,
// from the set of vertices not yet included in MST
int minKey(long long int key[], bool mstSet[], int V) {
    long long int min = LLONG_MAX;
    int min_index;

    for (int v = 0; v < V; v++)
        if (mstSet[v] == false && key[v] < min)
            min = key[v], min_index = v;

    return min_index;
}

// Function to print the constructed MST stored in parent[]
void printMST(int parent[], vector<tuple<long long int, long long int, long long int>> graph) {
    printf("Edge   Weight\n");
    for (int i = 1; i < graph.size(); i++) {
        printf("%d - %lld    %lld \n", parent[i], get<0>(graph[i]), get<2>(graph[i]));
    }
}

// Function to construct and print MST for a graph represented using adjacency matrix representation
void primMST(vector<tuple<long long int, long long int, long long int>> graph, int V) {
    int parent[V]; // Array to store constructed MST
    long long int key[V];    // Key values used to pick minimum weight edge in cut
    bool mstSet[V]; // To represent set of vertices not yet included in MST

    // Initialize all keys as INFINITE
    for (int i = 0; i < V; i++)
        key[i] = LLONG_MAX, mstSet[i] = false;

    // Always include first  vertex in MST.
    key[0] = 0;     // Make key 0 so that this vertex is picked as first vertex
    parent[0] = -1; // First node is always root of MST

    // The MST will have V vertices
    for (int count = 0; count < V - 1; count++) {
        // Pick the minimum key vertex from the set of vertices not yet included in MST
        int u = minKey(key, mstSet, V);

        // Add the picked vertex to the MST Set
        mstSet[u] = true;

        // Update key value and parent index of the adjacent vertices of the picked vertex.
        // Consider only those vertices which are not yet included in MST
        for (int i = 0; i < graph.size(); i++) {
            long long int node1 = get<0>(graph[i]);
            long long int node2 = get<1>(graph[i]);
            long long int weight = get<2>(graph[i]);

            if ((node1 == u || node2 == u) && mstSet[node1] != mstSet[node2]) {
                int v = (node1 == u) ? node2 : node1;
                if (key[v] > weight) {
                    parent[v] = u;
                    key[v] = weight;
                }
            }
        }
    }

    // Print the constructed MST
    printMST(parent, graph);
}

int main() {
    int V = 10; // Number of vertices
    vector<tuple<long long int, long long int, long long int>> graph = generate(V);

    for(long long int i = 0 ; i < graph.size() ; i++) {
        long long int node1 = get<0>(graph[i]);
        long long int node2 = get<1>(graph[i]);
        long long int weight = get<2>(graph[i]);        
        cout << node1 << " " << node2 << " " << weight << "\n";
    }

    printf("\n\n");

    // Print the solution
    primMST(graph, V);
    return 0;
}
