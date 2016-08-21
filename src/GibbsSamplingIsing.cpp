// Gibbs sampling on a synthetic Ising model.
// See http://arxiv.org/pdf/1602.07415v2.pdf for details.
#include <iostream>
#include <omp.h>
#include <map>
#include <vector>
#include <limits>

using namespace std;

#define N 1000                // Number of vertices
#define DELTA 3               // Maximum degree of vertices
#define BETA .2               // Inverse temperature
#define PRIOR_WEIGHTS 0       // Prior weights Ex

#define MAX_EDGES (N*N)
#define MAX_EDGE_INSERTION_TRIES (N*N)

typedef map<int, vector<int> > Graph;

void PrintGraph(Graph &g) {
    for (int i = 0; i < N; i++) {
	cout << i << ": ";
	for (int j = 0; j < g[i].size(); j++) {
	    if (j != 0) cout << ", ";
	    cout << g[i][j];
	}
	cout << endl;
    }
}

void PrintGraphStatistics(Graph &g) {
    double min_degree = DELTA+1;
    double max_degree = 0;
    double avg_degree = 0;
    for (int i = 0; i < N; i++) {
	min_degree = min(min_degree, (double)g[i].size());
	max_degree = max(max_degree, (double)g[i].size());
	avg_degree += g[i].size();
    }
    avg_degree /= N;
    printf("Graph statistics:\n");
    printf("Min Degree: %lf\n", min_degree);
    printf("Max Degree: %lf\n", max_degree);
    printf("Avg Degree: %lf\n", avg_degree);
}

// Initialize an empty graph g with a
// synthetic ising graph.
Graph GenerateIsingModelGraph() {
    Graph g;

    // Initialize vertices.
    for (int i = 0; i < N; i++) {
	g[i] = vector<int>();
    }

    // Create random edges but make sure
    // the Delta limit is not exceeded.
    for (int i = 0; i < MAX_EDGES; i++) {
	int random_vertex_1 = rand() % N;
	int random_vertex_2 = rand() % N;
	int n_tries = 0;

	while (random_vertex_1 == random_vertex_2 ||
	       g[random_vertex_1].size() >= DELTA ||
	       g[random_vertex_2].size() >= DELTA) {
	    random_vertex_1 = rand() % N;
	    random_vertex_2 = rand() % N;

	    // Break if could not find valid edge to insert.
	    if (n_tries++ >= MAX_EDGE_INSERTION_TRIES)
		return g;
	}

	g[random_vertex_1].push_back(random_vertex_2);
	g[random_vertex_2].push_back(random_vertex_1);
    }
    return g;
}

int main(int argc, char *argv[]) {
    Graph g = GenerateIsingModelGraph();
    PrintGraphStatistics(g);
}
