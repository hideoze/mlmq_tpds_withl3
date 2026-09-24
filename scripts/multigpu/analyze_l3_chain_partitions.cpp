// Read-only statistics from the exact runtime chain-index builder.
#include "../../SSSP/l3/l3_chain_partition.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <map>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

struct Mapping {
    int fd = -1;
    size_t bytes = 0;
    const unsigned char *data = nullptr;
    ~Mapping() {
        if (data && data != MAP_FAILED) munmap(const_cast<unsigned char *>(data), bytes);
        if (fd >= 0) close(fd);
    }
};

static void print_index(const char *name, int begin, int end,
                        const l3_chain_partition_index &index, double build_ms) {
    std::map<int, long long> histogram;
    int longest = 0;
    for (size_t base = 0; base < index.nodes.size();) {
        const int length = index.nodes[base].length;
        if (length < 4 || base + length > index.nodes.size())
            throw std::runtime_error("invalid runtime index record");
        const int interiors = length - 2;
        ++histogram[interiors];
        longest = std::max(longest, interiors);
        base += length;
    }
    long long route_members = 0;
    for (uint64_t route : index.route) route_members += route != 0;
    if (route_members != index.interiors)
        throw std::runtime_error("route/interior accounting mismatch");
    const size_t route_bytes = index.route.size() * sizeof(uint64_t);
    const size_t node_bytes = index.nodes.size() * sizeof(l3_chain_partition_node);
    std::printf(
        "\"%s\":{\"begin\":%d,\"end\":%d,\"vertices\":%d,"
        "\"segments\":%d,\"interiors\":%d,\"coverage\":%.12f,"
        "\"overflow_segments\":%d,\"longest_interiors\":%d,"
        "\"route_bytes\":%zu,\"node_bytes\":%zu,\"index_bytes\":%zu,"
        "\"diagnostic_visit_bytes\":%zu,"
        "\"build_ms\":%.6f,\"histogram\":{",
        name, begin, end, end - begin, index.segments, index.interiors,
        double(index.interiors) / double(end - begin), index.overflow, longest,
        route_bytes, node_bytes, route_bytes + node_bytes,
        index.nodes.size() * sizeof(unsigned), build_ms);
    bool first = true;
    for (const auto &[length, count] : histogram) {
        std::printf("%s\"%d\":%lld", first ? "" : ",", length, count);
        first = false;
    }
    std::printf("}}");
}

int main(int argc, char **argv) try {
    if (argc != 3)
        throw std::runtime_error("usage: analyze_l3_chain_partitions INPUT.gr CUT_PERCENT");
    char *cut_end = nullptr;
    const long cut_percent = std::strtol(argv[2], &cut_end, 10);
    if (!cut_end || *cut_end || cut_percent < 1 || cut_percent > 99)
        throw std::runtime_error("CUT_PERCENT must be in [1,99]");

    Mapping mapping;
    mapping.fd = open(argv[1], O_RDONLY);
    if (mapping.fd < 0) throw std::runtime_error("open failed");
    struct stat status {};
    if (fstat(mapping.fd, &status) || status.st_size < 32)
        throw std::runtime_error("invalid file size");
    mapping.bytes = static_cast<size_t>(status.st_size);
    mapping.data = static_cast<const unsigned char *>(
        mmap(nullptr, mapping.bytes, PROT_READ, MAP_PRIVATE, mapping.fd, 0));
    if (mapping.data == MAP_FAILED) throw std::runtime_error("mmap failed");

    const auto *header = reinterpret_cast<const uint64_t *>(mapping.data);
    if (header[0] != 1 || header[1] != 4 || header[2] == 0 ||
        header[2] >= INT_MAX || header[3] >= INT_MAX)
        throw std::runtime_error("unsupported GR header");
    const int vertices = static_cast<int>(header[2]);
    const int edges = static_cast<int>(header[3]);
    const size_t destination_offset = 32 + 8ull * vertices;
    const size_t destination_bytes = 4ull * edges;
    const size_t destination_padding = 4ull * (edges & 1);
    const size_t weight_offset = destination_offset + destination_bytes + destination_padding;
    const size_t expected_bytes = weight_offset + 4ull * edges;
    if (mapping.bytes != expected_bytes)
        throw std::runtime_error("GR length mismatch");

    const auto *row_ends = reinterpret_cast<const uint64_t *>(mapping.data + 32);
    const auto *destinations = reinterpret_cast<const int *>(
        mapping.data + destination_offset);
    const auto *weights = reinterpret_cast<const int *>(mapping.data + weight_offset);
    std::vector<int> rows(vertices + 1, 0);
    for (int vertex = 0; vertex < vertices; ++vertex) {
        if (row_ends[vertex] > static_cast<uint64_t>(edges) ||
            row_ends[vertex] < static_cast<uint64_t>(rows[vertex]))
            throw std::runtime_error("invalid CSR row offsets");
        rows[vertex + 1] = static_cast<int>(row_ends[vertex]);
    }
    if (rows.back() != edges) throw std::runtime_error("CSR does not end at edge count");

    const int cut = std::max(1, std::min(vertices - 1,
        static_cast<int>(static_cast<long long>(vertices) * cut_percent / 100)));
    l3_chain_partition_index first;
    l3_chain_partition_index second;
    const auto start_first = std::chrono::steady_clock::now();
    first.build(vertices, rows.data(), destinations, weights, 0, cut);
    const auto end_first = std::chrono::steady_clock::now();
    second.build(vertices, rows.data(), destinations, weights, cut, vertices);
    const auto end_second = std::chrono::steady_clock::now();
    const double first_ms = std::chrono::duration<double, std::milli>(
        end_first - start_first).count();
    const double second_ms = std::chrono::duration<double, std::milli>(
        end_second - end_first).count();

    std::printf(
        "{\"schema\":1,\"input\":\"%s\",\"vertices\":%d,"
        "\"edges\":%d,\"cut_percent\":%ld,\"cut\":%d,\"owners\":{",
        argv[1], vertices, edges, cut_percent, cut);
    print_index("gpu0", 0, cut, first, first_ms);
    std::printf(",");
    print_index("gpu1", cut, vertices, second, second_ms);
    const long long interiors = static_cast<long long>(first.interiors) + second.interiors;
    const long long segments = static_cast<long long>(first.segments) + second.segments;
    const long long bytes =
        static_cast<long long>(first.route.size() + second.route.size()) * sizeof(uint64_t) +
        static_cast<long long>(first.nodes.size() + second.nodes.size()) *
            sizeof(l3_chain_partition_node);
    const long long diagnostic_visit_bytes =
        static_cast<long long>(first.nodes.size() + second.nodes.size()) *
            sizeof(unsigned);
    std::printf(
        "},\"combined\":{\"segments\":%lld,\"interiors\":%lld,"
        "\"coverage\":%.12f,\"index_bytes\":%lld,"
        "\"diagnostic_visit_bytes\":%lld,\"build_ms\":%.6f}}\n",
        segments, interiors, double(interiors) / double(vertices), bytes,
        diagnostic_visit_bytes, first_ms + second_ms);
    return 0;
} catch (const std::exception &error) {
    std::fprintf(stderr, "CHAIN_ANALYZE_ERROR %s\n", error.what());
    return 2;
}
