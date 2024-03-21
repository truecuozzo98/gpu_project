#include <bits/stdc++.h>
#include "random_graph_generator.h"
#define MAX_EDGE_WEIGHT 100
#define MAX_NODES 100000000
#define MIN_EDGE_WEIGHT 10

using namespace std;

std::vector<std::vector<std::pair<long long int, long long int>>> generate(int nodes) {
    vector<vector<pair<long long int, long long int>>> adjacency(nodes);

    long long int extra_edges = ((nodes - 1) * nodes)/2 - (nodes-1);
    extra_edges = rand() % (extra_edges + 1);

    if(nodes - 1 + extra_edges > MAX_NODES){
        long long int difference = MAX_NODES - (nodes - 1);
        extra_edges = rand() % (difference + 1);
    }

    vector<long long int> graph(nodes);
    
    for(long long int i = 0; i < nodes; ++i){
        graph[i] = i;
    }   

    random_shuffle(graph.begin(),graph.end());

    set<pair<long long int, long long int> > present_edge;

    for(long long int i = 1; i < nodes; ++i){
        long long int add = rand() % i;
        long long int weight = rand() % MAX_EDGE_WEIGHT;
        adjacency[graph[i]].push_back(make_pair(graph[add], weight));
        adjacency[graph[add]].push_back(make_pair(graph[i], weight));
        present_edge.insert(make_pair(min(graph[add], graph[i]), max(graph[add], graph[i])));
    }

    for(long long int i = 1; i <= extra_edges; ++i){
        long long int weight = rand() % MAX_EDGE_WEIGHT;
        while(1){
            long long int node1 = rand() % nodes;
            long long int node2 = rand() % nodes;
            if(node1 == node2) continue;
            if(present_edge.find(make_pair(min(node1, node2), max(node1, node2))) == present_edge.end()){
                adjacency[node1].push_back(make_pair(node2, weight));
                adjacency[node2].push_back(make_pair(node1, weight));
                present_edge.insert(make_pair(min(node1, node2), max(node1, node2)));
                break;
            }
        }
    }
    return adjacency;
}