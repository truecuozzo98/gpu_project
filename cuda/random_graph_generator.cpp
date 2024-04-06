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
/*
    //write to file
    ofstream graph_file("graph.txt");
    for(long long int i = 0; i < NODES; ++i){
        for(long long int j = 0; j < adjacency[i].size(); ++j){
            //if(i < adjacency[i][j].first){
                graph_file<<i<<" "<<adjacency[i][j].first<<" "<<adjacency[i][j].second<<"\n";
            //}
        }
    }
    graph_file.close();
*/
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

/*
int main() {
    Graph graph = generate();
    int edges = tot_edges(NODES, graph);
    printf("Generated graph of %d vertices and %d edges:\n", NODES, edges);
    print_graph(graph);
    return 0;
}
*/