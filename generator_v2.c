#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_NODES 1000
#define MAX_EDGE_WEIGHT 100
#define MIN_EDGE_WEIGHT 1

struct Edge {
    int node1;
    int node2;
    int weight;

};

void generate(int nodes) {
    struct Edge* adjacency[MAX_NODES];

    for (int i = 0; i < nodes; ++i) {
        adjacency[i] = (struct Edge*)malloc(nodes * sizeof(struct Edge));
    }

    long long int extra_edges = ((nodes - 1) * nodes) / 2 - (nodes - 1);
    extra_edges = rand() % (extra_edges + 1);

    if (nodes - 1 + extra_edges > MAX_NODES) {
        long long int difference = MAX_NODES - (nodes - 1);
        extra_edges = rand() % (difference + 1);
    }

    long long int graph[MAX_NODES];

    for (long long int i = 0; i < nodes; ++i) {
        graph[i] = i;
    }

    for (long long int i = 1; i < nodes; ++i) {
        long long int add = rand() % i;
        long long int weight = rand() % MAX_EDGE_WEIGHT + MIN_EDGE_WEIGHT;
        adjacency[graph[i]][i].node = graph[add];
        adjacency[graph[i]][i].weight = weight;
        adjacency[graph[add]][i].node = graph[i];
        adjacency[graph[add]][i].weight = weight;
    }

    for (long long int i = 1; i <= extra_edges; ++i) {
        long long int weight = rand() % MAX_EDGE_WEIGHT + MIN_EDGE_WEIGHT;
        while (1) {
            long long int node1 = rand() % nodes;
            long long int node2 = rand() % nodes;
            if (node1 == node2) {
                continue;
            }
            if (adjacency[node1][node2].node == -1) {
                adjacency[node1][node2].node = node2;
                adjacency[node1][node2].weight = weight;
                adjacency[node2][node1].node = node1;
                adjacency[node2][node1].weight = weight;
                break;
            }
        }
    }

    //freopen("graph", "w", stdout);

    printf("%d %lld\n", nodes, nodes - 1 + extra_edges);

    for (long long int i = 0; i < nodes; ++i) {
        for (long long int j = 0; j < nodes; ++j) {
            if (adjacency[i][j].node != -1 && i < adjacency[i][j].node) {
                printf("%lld %lld %lld\n", i, adjacency[i][j].node, adjacency[i][j].weight);
            }
        }
    }

    //fclose(stdout);
}

int main() {
    srand(time(NULL));
    int nodes = 10; // Number of nodes in the graph
    generate(nodes);
    return 0;
}
