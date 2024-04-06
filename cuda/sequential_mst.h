using namespace std;

typedef pair<int, int> Edge; // Define edge type
typedef vector<vector<Edge>> Graph;

Graph sequential_prim_MST(const Graph& graph, int n);