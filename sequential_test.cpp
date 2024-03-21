#include <bits/stdc++.h>
#include "random_graph_generator.h"

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