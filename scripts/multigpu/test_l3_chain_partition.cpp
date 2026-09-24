#include "../../SSSP/l3/l3_chain_partition.h"

#include <cassert>
#include <climits>
#include <cstdio>
#include <functional>
#include <queue>
#include <random>
#include <utility>
#include <vector>

using Adj = std::vector<std::vector<std::pair<int, int>>>;

struct CSR {
    std::vector<int> row{0};
    std::vector<int> dst;
    std::vector<int> weight;

    explicit CSR(const Adj &adjacency) {
        for (const auto &edges : adjacency) {
            for (const auto &edge : edges) {
                dst.push_back(edge.first);
                weight.push_back(edge.second);
            }
            row.push_back(static_cast<int>(dst.size()));
        }
    }
};

static std::vector<long long> solve(
    const CSR &graph, int source,
    const l3_chain_partition_index *index = nullptr) {
    const int n = static_cast<int>(graph.row.size()) - 1;
    std::vector<long long> distance(n, INT_MAX);
    using Item = std::pair<long long, int>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> queue;
    distance[source] = 0;
    queue.push({0, source});
    while (!queue.empty()) {
        const auto [key, vertex] = queue.top();
        queue.pop();
        if (key != distance[vertex]) continue;
        const uint64_t route = index ? index->route[vertex + 1] : 0;
        if (route) {
            const int base = static_cast<int>(route >> 32) - 1;
            const int origin_index = static_cast<int>(uint32_t(route));
            const auto origin = index->nodes[origin_index];
            const int length = index->nodes[base].length;
            for (int i = base; i < base + length; ++i) {
                const auto target = index->nodes[i];
                const long long candidate = key +
                    (i > origin_index ? target.forward - origin.forward
                                      : origin.reverse - target.reverse);
                const int target_vertex = target.id - 1;
                if (candidate < distance[target_vertex]) {
                    distance[target_vertex] = candidate;
                    if (i == base || i == base + length - 1)
                        queue.push({candidate, target_vertex});
                }
            }
        } else {
            for (int edge = graph.row[vertex]; edge < graph.row[vertex + 1]; ++edge) {
                const long long candidate = key + graph.weight[edge];
                if (candidate < distance[graph.dst[edge]]) {
                    distance[graph.dst[edge]] = candidate;
                    queue.push({candidate, graph.dst[edge]});
                }
            }
        }
    }
    return distance;
}

static void link(Adj &adjacency, int first, int second,
                 int forward, int reverse) {
    adjacency[first].push_back({second, forward});
    adjacency[second].push_back({first, reverse});
}

static int verify(const Adj &adjacency, int begin = 0, int end = -1) {
    const CSR graph(adjacency);
    const int n = static_cast<int>(adjacency.size());
    if (end < 0) end = n;
    l3_chain_partition_index index;
    index.build(n, graph.row.data(), graph.dst.data(), graph.weight.data(),
                begin, end);
    if (begin == 0 && end == n) {
        for (int source = 0; source < n; ++source)
            assert(solve(graph, source) == solve(graph, source, &index));
    }
    const auto route = index.route;
    const auto nodes = index.nodes;
    index.build(n, graph.row.data(), graph.dst.data(), graph.weight.data(),
                begin, end);
    assert(index.route == route);
    assert(index.nodes.size() == nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
        assert(index.nodes[i].id == nodes[i].id);
        assert(index.nodes[i].forward == nodes[i].forward);
        assert(index.nodes[i].reverse == nodes[i].reverse);
        assert(index.nodes[i].length == nodes[i].length);
    }
    for (int vertex = begin; vertex < end; ++vertex) {
        const int route_index = vertex + 1 - begin;
        if (!index.route[route_index]) continue;
        const int base = static_cast<int>(index.route[route_index] >> 32) - 1;
        const int origin_index = static_cast<int>(
            uint32_t(index.route[route_index]));
        assert(index.nodes[origin_index].id == vertex + 1);
        assert(graph.row[vertex + 1] - graph.row[vertex] == 2);
        const auto actual = solve(graph, vertex);
        const auto origin = index.nodes[origin_index];
        for (int i = base; i < base + index.nodes[base].length; ++i) {
            const auto target = index.nodes[i];
            const int cost = i > origin_index
                ? target.forward - origin.forward
                : origin.reverse - target.reverse;
            assert(actual[target.id - 1] <= cost);
        }
    }
    return index.interiors;
}

static void parallel_edge_must_fall_back() {
    // Vertex 1 has two distinct outgoing neighbours but both incoming edges
    // originate at vertex 0.  The old aggregate reverse count accepted it and
    // later threw on the missing 2->1 edge.  It is not a strict chain interior.
    Adj adjacency(4);
    adjacency[0] = {{1, 2}, {1, 3}};
    adjacency[1] = {{0, 5}, {2, 7}};
    adjacency[2] = {{3, 11}};
    adjacency[3] = {{2, 13}};
    const CSR graph(adjacency);
    l3_chain_partition_index index;
    index.build(4, graph.row.data(), graph.dst.data(), graph.weight.data(), 0, 4);
    assert(index.segments == 0);
    assert(index.interiors == 0);
    for (uint64_t route : index.route) assert(route == 0);
}

static void offset_partition_route_contract() {
    Adj adjacency(8);
    link(adjacency, 4, 5, 5, 7);
    link(adjacency, 5, 6, 0, 11);
    link(adjacency, 6, 7, 13, 17);
    const CSR graph(adjacency);
    l3_chain_partition_index index;
    index.build(8, graph.row.data(), graph.dst.data(), graph.weight.data(), 4, 8);
    assert(index.segments == 1);
    assert(index.interiors == 2);
    assert(index.route[5 + 1 - 4] != 0);
    assert(index.route[6 + 1 - 4] != 0);
    assert(index.route[4 + 1 - 4] == 0);
    assert(index.route[7 + 1 - 4] == 0);
}

int main() {
    int fixtures = 0;
    int covered = 0;
    for (int mode = 0; mode < 4; ++mode) {
        Adj adjacency(131);
        for (int vertex = 0; vertex < 130; ++vertex) {
            const int forward = mode == 0 ? 0 : mode == 3 ? INT_MAX / 2 : 1;
            const int reverse = mode == 0 ? 0 : mode == 3 ? INT_MAX / 2 : 7;
            link(adjacency, vertex, vertex + 1, forward, reverse);
        }
        if (mode == 2) {
            adjacency[45].push_back({45, 0});
            adjacency[75].push_back({76, 3});
            adjacency[10].push_back({110, 1000});
        }
        covered += verify(adjacency);
        ++fixtures;
    }

    std::mt19937 random(230);
    for (int trial = 0; trial < 200; ++trial) {
        const int n = 3 + random() % 40;
        Adj adjacency(n);
        for (int vertex = 0; vertex < n - 1; ++vertex) {
            if (random() % 6)
                link(adjacency, vertex, vertex + 1,
                     random() % 100, random() % 100);
        }
        for (int edge = 0; edge < n / 6; ++edge) {
            const int source = random() % n;
            const int target = random() % n;
            adjacency[source].push_back({target, static_cast<int>(random() % 100)});
        }
        covered += verify(adjacency);
        ++fixtures;
    }

    parallel_edge_must_fall_back();
    offset_partition_route_contract();
    assert(covered > 0);
    std::printf(
        "L3_CHAIN_PARTITION_CPU fixtures=%d covered=%d "
        "all_sources/asymmetric/zero/overflow/parallel_fallback/offset/reset PASS\n",
        fixtures + 2, covered);
    return 0;
}
