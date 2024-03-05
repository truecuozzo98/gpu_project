#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_NODES 100

// Function to generate a random adjacency matrix for a graph with 'numNodes' nodes
void generateRandomAdjMatrix(int numNodes, int adjMatrix[MAX_NODES][MAX_NODES]) {
    // Seed the random number generator
    srand(time(NULL));

    // Generate random adjacency matrix
    for (int i = 0; i < numNodes; i++) {
        for (int j = 0; j < numNodes; j++) {
            // Assign a random value of 0 or 1 to the matrix cell
            adjMatrix[i][j] = rand() % 2;
        }
    }
}

// Function to print the adjacency matrix
void printAdjMatrix(int numNodes, int adjMatrix[MAX_NODES][MAX_NODES]) {
    printf("Adjacency Matrix:\n");
    for (int i = 0; i < numNodes; i++) {
        for (int j = 0; j < numNodes; j++) {
            printf("%d ", adjMatrix[i][j]);
        }
        printf("\n");
    }
}

int main() {
    int numNodes;
    printf("Enter the number of nodes in the graph: ");
    scanf("%d", &numNodes);

    if (numNodes <= 0 || numNodes > MAX_NODES) {
        printf("Number of nodes should be between 1 and %d\n", MAX_NODES);
        return 1;
    }

    int adjMatrix[MAX_NODES][MAX_NODES];

    // Generate random adjacency matrix
    generateRandomAdjMatrix(numNodes, adjMatrix);

    // Print the adjacency matrix
    printAdjMatrix(numNodes, adjMatrix);

    return 0;
}
