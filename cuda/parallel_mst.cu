#include <iostream>
#include <random>
#include <vector>
#include <set>
#define MAX_EDGE_WEIGHT 100
#define MAX_NODES 100000000
#define MIN_EDGE_WEIGHT 10

using namespace std;

typedef long long int ll;
typedef pair<ll, ll> Edge; // Define edge type
typedef vector<vector<Edge>> Graph; // Define graph type

void printGraph(int n, const Graph& graph) {
    for (int u = 0; u < n; ++u) {
        for (const auto& neighbor : graph[u]) {
            ll v = neighbor.first;
            ll weight = neighbor.second;
            cout << u << " - " << v << " : " << weight << endl;
        }
    }
    printf("\n");
}

Graph generate(int nodes) {
    Graph adjacency(nodes);

    ll extra_edges = ((nodes - 1) * nodes)/2 - (nodes-1);
    extra_edges = rand() % (extra_edges + 1);

    if(nodes - 1 + extra_edges > MAX_NODES){
        ll difference = MAX_NODES - (nodes - 1);
        extra_edges = rand() % (difference + 1);
    }

    vector<ll> graph(nodes);

    for(ll i = 0; i < nodes; ++i){
        graph[i] = i;
    }

    shuffle(graph.begin(),graph.end(), std::mt19937(std::random_device()()));

    set<Edge> present_edge;

    for(ll i = 1; i < nodes; ++i){
        ll add = rand() % i;
        ll weight = rand() % MAX_EDGE_WEIGHT;
        adjacency[graph[i]].emplace_back(graph[add], weight);
        adjacency[graph[add]].emplace_back(graph[i], weight);
        present_edge.insert(make_pair(min(graph[add], graph[i]), max(graph[add], graph[i])));
    }

    for(ll i = 1; i <= extra_edges; ++i){
        ll weight = rand() % MAX_EDGE_WEIGHT;
        while(true){
            ll node1 = rand() % nodes;
            ll node2 = rand() % nodes;
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

int main() {
    int numVertices = 10;
    int source = 0;
    Graph graph = generate(numVertices);
    int edges = totEdges(numVertices, graph);

    printf("Generated graph of %d vertices and %d edges:\n", numVertices, edges);
    printGraph(numVertices, graph);

    return 0;
}