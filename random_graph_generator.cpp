#include <bits/stdc++.h>
#define MAX_EDGE_WEIGHT 100
#define MAX_NODES 100000000
#define MIN_EDGE_WEIGHT 10

using namespace std;

std::vector<std::tuple<long long, long long, long long>> generate(int nodes) {
    vector<vector<tuple<long long int, long long int, long long int>>> adjacency(nodes);

    long long int extra_edges = ((nodes - 1) * nodes) / 2 - (nodes - 1);
    extra_edges = rand() % (extra_edges + 1);

    if (nodes - 1 + extra_edges > MAX_NODES) {
        long long int difference = MAX_NODES - (nodes - 1);
        extra_edges = rand() % (difference + 1);
    }

    vector<long long int> graph(nodes);

    for (long long int i = 0; i < nodes; ++i) {
        graph[i] = i;
    }

    random_shuffle(graph.begin(), graph.end());

    set<pair<long long int, long long int>> present_edge;

    for (long long int i = 1; i < nodes; ++i) {
        long long int add = rand() % i;
        long long int weight = rand() % MAX_EDGE_WEIGHT + 1;
        adjacency[graph[i]].push_back(make_tuple(graph[i], graph[add], weight));
        adjacency[graph[add]].push_back(make_tuple(graph[add], graph[i], weight));
        present_edge.insert(make_pair(min(graph[add], graph[i]), max(graph[add], graph[i])));
    }

    for (long long int i = 1; i <= extra_edges; ++i) {
        long long int weight = rand() % MAX_EDGE_WEIGHT;
        while (1) {
            long long int node1 = rand() % nodes;
            long long int node2 = rand() % nodes;
            if (node1 == node2) {
                continue;
            }
            if (present_edge.find(make_pair(min(node1, node2), max(node1, node2))) == present_edge.end()) {
                adjacency[node1].push_back(make_tuple(node1, node2, weight));
                adjacency[node2].push_back(make_tuple(node2, node1, weight));
                present_edge.insert(make_pair(min(node1, node2), max(node1, node2)));
                break;
            }
        }
    }

    vector<tuple<long long int, long long int, long long int>> edges;
    for (long long int i = 0; i < nodes; ++i) {
        for (long long int j = 0; j < adjacency[i].size(); ++j) {
            long long int node1 = get<0>(adjacency[i][j]);
            long long int node2 = get<1>(adjacency[i][j]);
            long long int weight = get<2>(adjacency[i][j]);
            edges.push_back(make_tuple(node1, node2, weight));
        }
    }

    return edges;
}

int main(){

    srand(time(NULL));

    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);

    long long int nodes = 10;
    vector<tuple<long long int, long long int, long long int>> graph = generate(nodes);

    for(long long int i = 0 ; i < graph.size() ; i++) {
        long long int node1 = get<0>(graph[i]);
        long long int node2 = get<1>(graph[i]);
        long long int weight = get<2>(graph[i]);        
        cout << node1 << " " << node2 << " " << weight << "\n";
    }
    
    return 0;
}