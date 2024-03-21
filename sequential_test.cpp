#include <bits/stdc++.h>
#include "random_graph_generator.h"

int main(){

    srand(time(NULL));

    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);

    long long int nodes = 10;
    vector<vector<pair<long long int, long long int>>> graph = generate(nodes);

    for(long long int i = 0; i < nodes; ++i){
        for(long long int j = 0; j < graph[i].size(); ++j){
            if(i < graph[i][j].first){
                cout<<i<<" "<<graph[i][j].first<<" "<<graph[i][j].second<<"\n";
            }
        }
    }
    
    return 0;
}