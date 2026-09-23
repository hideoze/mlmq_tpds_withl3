// Host-only int64 SSSP oracle for the Galois v1 GR layout used by MLMQ.
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

struct Graph {
    int fd = -1;
    size_t bytes = 0;
    const char *mapping = nullptr;
    int n = 0;
    int m = 0;
    const uint64_t *ends = nullptr;
    const uint32_t *destinations = nullptr;
    const int32_t *weights = nullptr;
    bool unit_weights = true;

    explicit Graph(const char *path) {
        fd = open(path, O_RDONLY);
        struct stat info{};
        if (fd < 0 || fstat(fd, &info) != 0) fail("open/stat");
        bytes = size_t(info.st_size);
        if (bytes < 32) fail("short header");
        mapping = static_cast<const char *>(mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0));
        if (mapping == MAP_FAILED) {
            mapping = nullptr;
            fail("mmap");
        }
        const auto *header = reinterpret_cast<const uint64_t *>(mapping);
        if (header[0] != 1 || header[1] != 4 || header[2] >= INT_MAX ||
            header[3] >= INT_MAX || header[2] == 0) {
            fail("unsupported header");
        }
        n = int(header[2]);
        m = int(header[3]);
        const uint64_t minimum = 32ull + 8ull * n + 4ull * m +
                                 4ull * (m & 1) + 4ull * m;
        if (minimum > bytes) fail("truncated payload");
        ends = reinterpret_cast<const uint64_t *>(mapping + 32);
        destinations = reinterpret_cast<const uint32_t *>(mapping + 32 + 8ull * n);
        weights = reinterpret_cast<const int32_t *>(
            mapping + 32 + 8ull * n + 4ull * m + 4ull * (m & 1));
        uint64_t previous = 0;
        for (int vertex = 0; vertex < n; ++vertex) {
            if (ends[vertex] < previous || ends[vertex] > uint64_t(m)) fail("invalid rows");
            previous = ends[vertex];
        }
        if (previous != uint64_t(m)) fail("last row does not end at edge count");
        for (int edge = 0; edge < m; ++edge) {
            if (destinations[edge] >= uint32_t(n) || weights[edge] < 0) {
                fail("invalid destination or negative weight");
            }
            unit_weights = unit_weights && weights[edge] == 1;
        }
    }

    ~Graph() {
        if (mapping) munmap(const_cast<char *>(mapping), bytes);
        if (fd >= 0) close(fd);
    }

    [[noreturn]] static void fail(const char *message) {
        throw std::runtime_error(message);
    }

    uint64_t begin(int vertex) const { return vertex == 0 ? 0 : ends[vertex - 1]; }
    uint64_t end(int vertex) const { return ends[vertex]; }
};

static std::vector<int64_t> solve(const Graph &graph, int source) {
    const int64_t infinity = std::numeric_limits<int64_t>::max();
    std::vector<int64_t> distance(graph.n, infinity);
    distance.at(source) = 0;
    if (graph.unit_weights) {
        std::queue<int> queue;
        queue.push(source);
        while (!queue.empty()) {
            const int vertex = queue.front();
            queue.pop();
            const int64_t candidate = distance[vertex] + 1;
            for (uint64_t edge = graph.begin(vertex); edge < graph.end(vertex); ++edge) {
                const int target = int(graph.destinations[edge]);
                if (distance[target] == infinity) {
                    distance[target] = candidate;
                    queue.push(target);
                }
            }
        }
        return distance;
    }

    using Entry = std::pair<int64_t, int>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> queue;
    queue.push({0, source});
    while (!queue.empty()) {
        const auto [cost, vertex] = queue.top();
        queue.pop();
        if (cost != distance[vertex]) continue;
        for (uint64_t edge = graph.begin(vertex); edge < graph.end(vertex); ++edge) {
            const int64_t weight = graph.weights[edge];
            if (cost > std::numeric_limits<int64_t>::max() - weight) {
                throw std::runtime_error("int64 candidate addition overflow");
            }
            const int64_t candidate = cost + weight;
            const int target = int(graph.destinations[edge]);
            if (candidate < distance[target]) {
                distance[target] = candidate;
                queue.push({candidate, target});
            }
        }
    }
    return distance;
}

template <typename T>
static void write_vector(const std::string &path, const std::vector<T> &values) {
    std::ifstream existing(path, std::ios::binary);
    if (existing.good()) throw std::runtime_error("output already exists: " + path);
    std::ofstream stream(path, std::ios::binary | std::ios::out);
    stream.write(reinterpret_cast<const char *>(values.data()),
                 std::streamsize(values.size() * sizeof(T)));
    stream.close();
    if (!stream) throw std::runtime_error("failed to write: " + path);
}

int main(int argc, char **argv) try {
    if (argc != 4 && argc != 5) {
        std::fprintf(stderr, "usage: %s INPUT_GR SOURCE OUTPUT_I64 [OUTPUT_I32]\n", argv[0]);
        return 2;
    }
    Graph graph(argv[1]);
    const long parsed = std::stol(argv[2]);
    if (parsed < 0 || parsed >= graph.n) throw std::runtime_error("source out of range");
    const int source = int(parsed);
    const auto distance = solve(graph, source);
    int64_t maximum = 0;
    uint64_t reached = 0;
    uint64_t over_int32 = 0;
    for (const int64_t value : distance) {
        if (value == std::numeric_limits<int64_t>::max()) continue;
        ++reached;
        maximum = std::max(maximum, value);
        if (value >= INT_MAX) ++over_int32;
    }
    write_vector(argv[3], distance);
    if (argc == 5 && over_int32 == 0) {
        std::vector<int32_t> narrow(graph.n, INT_MAX);
        for (int vertex = 0; vertex < graph.n; ++vertex) {
            if (distance[vertex] != std::numeric_limits<int64_t>::max()) {
                narrow[vertex] = int32_t(distance[vertex]);
            }
        }
        write_vector(argv[4], narrow);
    }
    std::printf(
        "ORACLE64 status=%s input=%s source=%d vertices=%d edges=%d algorithm=%s "
        "reached=%llu unreachable=%llu max_distance=%lld over_int32=%llu "
        "output_i64=%s output_i32=%s\n",
        (argc == 5 && over_int32 != 0) ? "INT32_UNREPRESENTABLE" : "PASS",
        argv[1], source, graph.n, graph.m, graph.unit_weights ? "BFS" : "DIJKSTRA",
        static_cast<unsigned long long>(reached),
        static_cast<unsigned long long>(uint64_t(graph.n) - reached),
        static_cast<long long>(maximum), static_cast<unsigned long long>(over_int32),
        argv[3], argc == 5 && over_int32 == 0 ? argv[4] : "NOT_WRITTEN");
    return argc == 5 && over_int32 != 0 ? 3 : 0;
} catch (const std::exception &error) {
    std::fprintf(stderr, "ORACLE64_FAIL message=%s\n", error.what());
    return 2;
}
