using namespace std;

typedef pair<int, int> Edge; // Define edge type
typedef vector<vector<Edge>> Graph;

Graph generate(int nodes);
void print_graph(const Graph& graph, int nodes);
int tot_edges(const Graph& graph, int n);
void adjacency_list_to_matrix(const Graph& adjList, int* adj_matrix, size_t n);
void print_adj_matrix(const int *adj_matrix, int nodes);
void print_distance_vector(vector<int> distance_vector);