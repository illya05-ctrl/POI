/*
 * ============================================================
 *  Parallel Tabu Search with Aspiration Criterion
 *  Problem : Graph Coloring Problem (GCP)
 *  Version : FINAL  —  OpenMP + MPI hybrid
 * ============================================================
 *
 *  Architecture:
 *    - Each MPI process runs an independent Tabu Search
 *    - OpenMP threads parallelize the neighbourhood scan
 *    - Every SYNC_INTERVAL iterations all ranks share best
 *
 *  Compile:
 *    mpicxx -O2 -std=c++17 -fopenmp -o tabu_gcp tabu_gcp_final.cpp
 *
 *  Run (examples):
 *    # 1 MPI rank, 4 OMP threads
 *    OMP_NUM_THREADS=4  mpirun -np 1 ./tabu_gcp rand 9
 *
 *    # 4 MPI ranks x 16 OMP threads = 64 cores
 *    OMP_NUM_THREADS=16 mpirun -np 4 ./tabu_gcp hard1000.col 22
 *
 *    # Benchmark mode (CSV output for speedup graphs)
 *    OMP_NUM_THREADS=8  mpirun -np 1 ./tabu_gcp hard1000.col 22 bench
 *
 *  Usage:
 *    ./tabu_gcp [file.col | rand] [k] [bench]
 *      file.col  : DIMACS .col graph file
 *      rand      : generate random graph (n=1000, p=0.03)
 *      k         : number of colours to try  (default 9)
 *      bench     : benchmark mode → CSV: cores,threads,ranks,conflicts,iters,time
 * ============================================================
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <climits>
#include <cassert>
#include <iomanip>
#include <omp.h>
#include <mpi.h>

/* ─── Graph ───────────────────────────────────────────────── */

struct Graph {
    int n, m;
    std::vector<std::vector<int>>  adj;
    std::vector<std::vector<bool>> mat;

    Graph() : n(0), m(0) {}
    Graph(int n) : n(n), m(0), adj(n), mat(n, std::vector<bool>(n, false)) {}

    void add_edge(int u, int v) {
        if (u == v || mat[u][v]) return;
        adj[u].push_back(v);
        adj[v].push_back(u);
        mat[u][v] = mat[v][u] = true;
        ++m;
    }
};

/* ─── Graph generators ────────────────────────────────────── */

Graph generate_random(int n, double p, unsigned seed = 42) {
    Graph g(n);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int u = 0; u < n; ++u)
        for (int v = u+1; v < n; ++v)
            if (dist(rng) < p) g.add_edge(u, v);
    return g;
}

Graph load_dimacs(const std::string& filename) {
    std::ifstream f(filename);
    if (!f) {
        std::cerr << "ERROR: cannot open " << filename << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int N = 0;
    std::string line;
    Graph* g = nullptr;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == 'c') continue;
        if (line[0] == 'p') {
            std::istringstream ss(line);
            std::string a, b; int M;
            ss >> a >> b >> N >> M;
            g = new Graph(N);
        }
        if (line[0] == 'e' && g) {
            std::istringstream ss(line);
            char ch; int u, v;
            ss >> ch >> u >> v;
            g->add_edge(u-1, v-1);
        }
    }
    if (!g) { std::cerr << "ERROR: invalid DIMACS file\n"; MPI_Abort(MPI_COMM_WORLD, 1); }
    return *g;
}

/* ─── Coloring solution ───────────────────────────────────── */

struct Solution {
    int k;
    std::vector<int> color;
    int conflicts;
    std::vector<std::vector<int>> nc;   // nc[v][c] = #neighbours of v with colour c

    Solution() : k(0), conflicts(0) {}
    Solution(int n, int k)
        : k(k), color(n, 0), conflicts(0), nc(n, std::vector<int>(k, 0)) {}
    Solution(const Solution&) = default;
    Solution& operator=(const Solution&) = default;
};

Solution random_init(const Graph& g, int k, unsigned seed) {
    Solution sol(g.n, k);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> cd(0, k-1);
    for (int v = 0; v < g.n; ++v) sol.color[v] = cd(rng);
    for (int v = 0; v < g.n; ++v)
        for (int u : g.adj[v]) sol.nc[v][sol.color[u]]++;
    sol.conflicts = 0;
    for (int v = 0; v < g.n; ++v) sol.conflicts += sol.nc[v][sol.color[v]];
    sol.conflicts /= 2;
    return sol;
}

inline int delta_cost(const Solution& sol, int v, int c) {
    return sol.nc[v][c] - sol.nc[v][sol.color[v]];
}

void apply_move(Solution& sol, const Graph& g, int v, int new_c) {
    int old_c = sol.color[v];
    if (old_c == new_c) return;
    sol.conflicts += sol.nc[v][new_c] - sol.nc[v][old_c];
    sol.color[v] = new_c;
    for (int u : g.adj[v]) { sol.nc[u][old_c]--; sol.nc[u][new_c]++; }
}

bool validate(const Graph& g, const std::vector<int>& color) {
    for (int v = 0; v < g.n; ++v)
        for (int u : g.adj[v])
            if (color[v] == color[u]) return false;
    return true;
}

/* ─── Parallel Tabu Search ────────────────────────────────── */

struct TabuResult {
    int best_conflicts;
    std::vector<int> best_color;
    long long iters;
    double time_sec;
};

TabuResult parallel_tabu(
    const Graph& g,
    int k,
    int max_iter,
    int tabu_tenure,
    int mpi_rank,
    int mpi_size,
    int omp_threads,
    int sync_interval = 500
) {
    auto t0 = std::chrono::high_resolution_clock::now();

    unsigned seed = 42 + (unsigned)mpi_rank * 1000;
    std::mt19937 rng_main(seed);

    Solution sol  = random_init(g, k, seed);
    Solution best = sol;
    int best_conf = sol.conflicts;

    // Tabu list: tabu[v][c] = iteration until move (v→c) is allowed again
    std::vector<std::vector<int>> tabu(g.n, std::vector<int>(k, 0));

    struct Move { int v, c, d; };
    std::vector<Move> thread_best(omp_threads, {-1, -1, INT_MAX});

    long long iter = 0;
    int no_improve = 0;
    const int MAX_NO_IMPROVE = max_iter / 4;

    std::vector<int> conf_verts;
    conf_verts.reserve(g.n);

    std::uniform_int_distribution<int> tnoise(0, std::max(1, tabu_tenure / 2));

    while (iter < max_iter && best_conf > 0) {

        /* Build list of conflicted vertices */
        conf_verts.clear();
        for (int v = 0; v < g.n; ++v)
            if (sol.nc[v][sol.color[v]] > 0)
                conf_verts.push_back(v);
        if (conf_verts.empty()) break;

        int n_conf = (int)conf_verts.size();
        for (auto& m : thread_best) { m.v = -1; m.c = -1; m.d = INT_MAX; }

        /* ── OpenMP: parallel neighbourhood scan ── */
        #pragma omp parallel num_threads(omp_threads)
        {
            int tid = omp_get_thread_num();
            Move local = {-1, -1, INT_MAX};

            #pragma omp for schedule(dynamic, 8) nowait
            for (int i = 0; i < n_conf; ++i) {
                int v = conf_verts[i];
                for (int c = 0; c < k; ++c) {
                    if (c == sol.color[v]) continue;
                    int d = delta_cost(sol, v, c);
                    bool is_tabu   = (tabu[v][c] > (int)iter);
                    bool aspiration = is_tabu && (sol.conflicts + d < best_conf);
                    if ((!is_tabu || aspiration) && d < local.d)
                        local = {v, c, d};
                }
            }
            thread_best[tid] = local;
        }

        /* Reduction: pick best move across threads */
        Move chosen = {-1, -1, INT_MAX};
        for (auto& m : thread_best)
            if (m.v != -1 && m.d < chosen.d) chosen = m;
        if (chosen.v == -1) break;

        /* Apply move */
        int old_c = sol.color[chosen.v];
        apply_move(sol, g, chosen.v, chosen.c);

        /* Update tabu list (forbid reverse) */
        tabu[chosen.v][old_c] = iter + tabu_tenure + tnoise(rng_main);

        /* Update local best */
        if (sol.conflicts < best_conf) {
            best_conf  = sol.conflicts;
            best       = sol;
            no_improve = 0;
            if (best_conf == 0) break;
        } else {
            ++no_improve;
        }

        /* Restart if stuck */
        if (no_improve >= MAX_NO_IMPROVE) {
            sol = random_init(g, k, iter + seed);
            no_improve = 0;
        }

        /* ── MPI sync: share global best ── */
        if (iter % sync_interval == 0 && mpi_size > 1) {
            struct { int conf; int rank; } loc = {best_conf, mpi_rank}, glob;
            MPI_Allreduce(&loc, &glob, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

            std::vector<int> shared(g.n);
            if (mpi_rank == glob.rank) shared = best.color;
            MPI_Bcast(shared.data(), g.n, MPI_INT, glob.rank, MPI_COMM_WORLD);

            if (glob.conf < best_conf) {
                best_conf      = glob.conf;
                best.color     = shared;
                best.conflicts = glob.conf;
                sol = random_init(g, k, iter + seed + 1);
                no_improve = 0;
            }
        }

        ++iter;
    }

    /* Final global sync */
    if (mpi_size > 1) {
        struct { int conf; int rank; } loc = {best_conf, mpi_rank}, glob;
        MPI_Allreduce(&loc, &glob, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);
        MPI_Bcast(best.color.data(), g.n, MPI_INT, glob.rank, MPI_COMM_WORLD);
        best_conf = glob.conf;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    return {best_conf, best.color, iter,
            std::chrono::duration<double>(t1 - t0).count()};
}

/* ─── Benchmark helper ────────────────────────────────────── */

void run_benchmark(const Graph& g, int k, int max_iter, int tenure,
                   int mpi_rank, int mpi_size, int omp_threads) {
    const int RUNS = 3;
    double   tt = 0;
    int      tc = 0;
    long long ti = 0;

    for (int r = 0; r < RUNS; ++r) {
        auto res = parallel_tabu(g, k, max_iter, tenure,
                                 mpi_rank, mpi_size, omp_threads);
        tt += res.time_sec;
        tc += res.best_conflicts;
        ti += res.iters;
    }

    if (mpi_rank == 0) {
        std::cout << (omp_threads * mpi_size) << ","
                  << omp_threads              << ","
                  << mpi_size                 << ","
                  << (tc / RUNS)              << ","
                  << (ti / RUNS)              << ","
                  << std::fixed << std::setprecision(4)
                  << (tt / RUNS)              << "\n";
    }
}

/* ─── Main ────────────────────────────────────────────────── */

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    int omp_threads = omp_get_max_threads();

    /* ── Build graph on rank 0, broadcast to others ── */
    Graph g;
    int gn = 0, gm = 0;
    std::vector<std::pair<int,int>> edges;

    if (mpi_rank == 0) {
        bool use_file = (argc >= 2 && std::string(argv[1]) != "rand");

        if (!use_file) {
            std::cout << "[Graph] Generating random (n=1000, p=0.03)\n";
            g = generate_random(1000, 0.03);
        } else {
            std::cout << "[Graph] Loading DIMACS: " << argv[1] << "\n";
            g = load_dimacs(argv[1]);
        }

        gn = g.n; gm = g.m;
        for (int u = 0; u < g.n; ++u)
            for (int v : g.adj[u])
                if (v > u) edges.push_back({u, v});

        std::cout << "[Graph] " << gn << " vertices, " << gm << " edges\n";
        std::cout << "[Parallel] MPI=" << mpi_size
                  << "  OMP=" << omp_threads
                  << "  total=" << (mpi_size * omp_threads) << " cores\n";
    }

    MPI_Bcast(&gn, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&gm, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (mpi_rank != 0) {
        g = Graph(gn);
        edges.resize(gm);
    }
    MPI_Bcast(edges.data(), gm * 2, MPI_INT, 0, MPI_COMM_WORLD);

    if (mpi_rank != 0)
        for (auto& e : edges) g.add_edge(e.first, e.second);

    /* ── Parameters ── */
    int  k     = (argc >= 3) ? std::atoi(argv[2]) : 9;
    bool bench = (argc >= 4 && std::string(argv[3]) == "bench");

    // Auto-tune tabu tenure based on graph size
    int tenure = std::max(7, (int)(0.6 * std::sqrt((double)g.n)));

    const int MAX_ITER = 300000;

    if (mpi_rank == 0) {
        std::cout << "[Params] k=" << k
                  << "  max_iter=" << MAX_ITER
                  << "  tabu_tenure=" << tenure << "\n\n";
    }

    /* ── Run ── */
    if (bench) {
        if (mpi_rank == 0)
            std::cout << "total_cores,omp_threads,mpi_ranks,conflicts,iters,time_sec\n";
        run_benchmark(g, k, MAX_ITER, tenure, mpi_rank, mpi_size, omp_threads);

    } else {
        TabuResult res = parallel_tabu(g, k, MAX_ITER, tenure,
                                       mpi_rank, mpi_size, omp_threads);
        if (mpi_rank == 0) {
            std::cout << "=== Result ===\n";
            std::cout << "k="         << k
                      << "  conflicts="  << res.best_conflicts
                      << "  iters="      << res.iters
                      << "  time="       << std::fixed << std::setprecision(3)
                      << res.time_sec    << "s\n";

            if (res.best_conflicts == 0) {
                assert(validate(g, res.best_color));
                std::cout << "FEASIBLE — valid " << k << "-coloring found!\n";
            } else {
                std::cout << "INFEASIBLE — " << res.best_conflicts
                          << " conflicts remain\n";
            }
        }
    }

    MPI_Finalize();
    return 0;
}