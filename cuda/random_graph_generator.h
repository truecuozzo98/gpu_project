using namespace std;

typedef pair<int, int> Edge; // Define edge type
typedef vector<vector<Edge>> Graph;

Graph generate(int nodes);
void print_graph(const Graph& graph, int nodes);
int tot_edges(const Graph& graph, int n);