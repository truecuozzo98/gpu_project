#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <vector>
#include <tuple>
#include "random_graph_generator.h"
#define V 10

using namespace std;

// Function to find the minimum weight vertex from the set of vertices not yet included in MST
int minWeightVertex(int n, long long int key[], int mstSet[]) {
    long long int minWeight = LLONG_MAX;
    int minWeightVertex;
    for (int v = 0; v < n; v++) {
        if (mstSet[v] == 0 && key[v] < minWeight) {
            minWeight = key[v];
            minWeightVertex = v;
        }
    }
    return minWeightVertex;
}

// Function to print MST using the parent array
void printMST(int n, int parent[], vector<vector<pair<long long int, long long int>>>& graph) {
    printf("Edge   Weight\n");
    for (int i = 1; i < n; i++) {
        int u = parent[i];
        int v = i;
        long long int weight = -1;
        // Find the weight of edge (u, v)
        for (auto& neighbor : graph[u]) {
            if (neighbor.first == v) {
                weight = neighbor.second;
                break;
            }
        }
        printf("%d - %d    %lld \n", u, v, weight);
    }
}


// Function to implement Prim's MST algorithm
void primMST(int n, vector<vector<pair<long long int, long long int>>>& graph) {
    int parent[n]; // Array to store constructed MST
    long long int key[n]; // Key values used to pick minimum weight edge in cut
    int mstSet[n]; // To represent set of vertices not yet included in MST

    // Initialize all keys as INFINITE
    for (int i = 0; i < n; i++) {
        key[i] = LLONG_MAX;
        mstSet[i] = 0;
    }

    // Always include first  vertex in MST.
    key[0] = 0; // Make key 0 so that this vertex is picked as first vertex
    parent[0] = -1; // First node is always root of MST

    // The MST will have n vertices
    for (int count = 0; count < n - 1; count++) {
        // Pick the minimum key vertex from the set of vertices not yet included in MST
        int u = minWeightVertex(n, key, mstSet);

        // Add the picked vertex to the MST set
        mstSet[u] = 1;

        // Update key value and parent index of the adjacent vertices of the picked vertex.
        // Consider only those vertices which are not yet included in MST
        for (auto& neighbor : graph[u]) {
            int v = neighbor.first;
            long long int weight = neighbor.second;
            if (mstSet[v] == 0 && weight < key[v]) {
                parent[v] = u;
                key[v] = weight;
            }
        }
    }

    // Print the constructed MST
    printMST(n, parent, graph);
}

int main() {
  
    vector<vector<pair<long long int, long long int>>> graph = generate(V);

    for(long long int i = 0; i < V; ++i){
        for(long long int j = 0; j < graph[i].size(); ++j){
            if(i < graph[i][j].first){
                cout<<i<<" "<<graph[i][j].first<<" "<<graph[i][j].second<<"\n";
            }
        }
    }
  
    printf("\n\n");

    // Print the MST using Prim's algorithm
    primMST(V, graph);

    return 0;
}