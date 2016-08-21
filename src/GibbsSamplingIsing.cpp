// Gibbs sampling on a synthetic Ising model.
// See http://arxiv.org/pdf/1602.07415v2.pdf for details.
// As in the paper,  assume prior weights B_x is 0.
#include <iostream>
#include <omp.h>
#include <map>
#include <vector>
#include <limits>
#include <math.h>

using namespace std;

#define HOGWILD 1
#define N_THREADS 1

#define N 1000                // Number of vertices
#define DELTA 3               // Maximum degree of vertices
#define BETA .2               // Inverse temperature
#define N_ITERATIONS 10000

#define MAX_EDGES (N*N)
#define MAX_EDGE_INSERTION_TRIES (N*N)

typedef map<int, vector<int> > Graph;

// Note that access pattern has form:
// [thread][batch][state index].
typedef vector<vector<vector<int> > > AccessPattern;

void PrintState(vector<int> &state) {
    // For conciseness, print -1 as 0.
    string state_string = "";
    for (int i = 0; i < state.size(); i++) {
	if (state[i] == 1) {
	    state_string += "1";
	}
	else if (state[i] == -1) {
	    state_string += "0";
	}
	else {
	    cout << "Something went wrong..." << endl;
	    exit(0);
	}
    }
    cout << state_string << endl;
}

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
Graph GenerateRandomIsingModelGraph() {
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

vector<int> GenerateIsingState() {
    vector<int> state(N);
    for (int i = 0; i < N; i++) {
	if (rand() % 2 == 0) {
	    state[i] = 1;
	}
	else {
	    state[i] = -1;
	}
    }
    return state;
}

int PartitionDatapointsForHogwild(Graph &g, vector<int> &state, AccessPattern &pattern) {
    pattern.resize(N_THREADS);
    int n_datapoints_per_thread = N / N_THREADS;
    for (int thread = 0; thread < N_THREADS; thread++) {
	pattern[thread].resize(1);
	int start = n_datapoints_per_thread * thread;
	int end = n_datapoints_per_thread * (thread+1);
	if (thread == N_THREADS-1) end = N;
	for (int index = start; index < end; index++) {
	    pattern[thread][0].push_back(index);
	}
    }
    return 1; // 1 batch for hogwild.
}

void UpdateState(Graph &g, vector<int> &state, int index) {
    int product_with_1 = 0;
    int product_with_neg_1 = 0;
    for (int i = 0; i < g[index].size(); i++) {
	product_with_1 += state[g[index][i]];
	product_with_neg_1 += state[g[index][i]] * -1;
    }

    double p1 = exp(BETA * (double)product_with_1);
    double p2 = exp(BETA * (double)product_with_neg_1);
    double prob_1 = p1 / (p1 + p2);
    double selection = ((double)rand() / (RAND_MAX));
    if (selection <= prob_1) {
	state[index] = 1;
    }
    else {
	state[index] = -1;
    }
}

int main(int argc, char *argv[]) {
    omp_set_num_threads(N_THREADS);

    // Generate graph.
    Graph g = GenerateRandomIsingModelGraph();
    PrintGraphStatistics(g);

    // Generate variables.
    vector<int> state = GenerateIsingState();

    // Access pattern partitions.
    // Of form [thread][batch][state to update].
    // Note that for hogwild, there will only be one batch
    AccessPattern access_pattern;
    int n_batches = 0;
    if (HOGWILD) {
	n_batches = PartitionDatapointsForHogwild(g, state, access_pattern);
    }

    for (int iter = 0; iter < N_ITERATIONS; iter++) {
#pragma omp parallel for num_threads(N_THREADS)
	for (int thread = 0; thread < N_THREADS; thread++) {
	    for (int batch = 0; batch < n_batches; batch++) {
		for (int to_update = 0; to_update < access_pattern[thread][batch].size(); to_update++) {
		    int index_to_update = access_pattern[thread][batch][to_update];
		    UpdateState(g, state, index_to_update);
		}
	    }
	}
	//PrintState(state);
    }
}
