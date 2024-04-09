#include <iostream>
#include <random>
#include <set>

#define MAX_EDGE_WEIGHT 100
#define MAX_NODES 100000000
#define MIN_EDGE_WEIGHT 10

using namespace std;
typedef pair<int, int> Edge; // Define edge type
typedef vector<vector<Edge>> Graph;

Graph generate(int nodes) {
    printf("generating graph...\n");

    // Use a random device to seed the random number engine
    random_device rd;
    // Use the Mersenne Twister engine for randomness
    mt19937 mt(rd());
    // Define the distribution for integers
    uniform_int_distribution<int> random_weight(MIN_EDGE_WEIGHT, MAX_EDGE_WEIGHT);
    int max_edges = ((nodes - 1) * nodes)/2 - (nodes-1);
    uniform_int_distribution<int> random_extra_edges((max_edges * 3)/4, max_edges);
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

void print_graph(const Graph& graph, int nodes) {
    for (int u = 0; u < nodes; ++u) {
        for (const auto& neighbor : graph[u]) {
            int v = neighbor.first;
            int weight = neighbor.second;
            cout << u << " " << v << " " << weight << endl;
        }
    }
    printf("\n");
}

int tot_edges(const Graph& graph, int n) {
    int count = 0;
    for (int u = 0; u < n; ++u) {
        count += (int) graph[u].size();
    }
    return count;
}

void adjacency_list_to_matrix(const Graph& adjList, int* adj_matrix, size_t n) {
    // Initialize the adjacency matrix with INF initially
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            adj_matrix[i * n + j] = INT_MAX;
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

void print_adj_matrix(const int *adj_matrix, int nodes) {
    for (size_t i = 0; i < nodes; ++i) {
        for (size_t j = 0; j < nodes; ++j) {
            int weight = adj_matrix[i * nodes + j];
            if (weight == INT_MAX) {
                cout << "INF\t";
            } else {
                cout << weight << "\t";
            }
        }
        cout << endl;
    }
}

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