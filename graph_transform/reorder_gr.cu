/*
  reorder_gr.cu

  顶点重排序工具（阶段 G / design_v3 §20）：按 BFS 序（root=0）重排 .gr 的顶点 id，
  使拓扑相邻的顶点获得连续 id，从而让 graph_partition 的连续 id 对半切得到低割边界，
  降低多卡 SSSP 的跨卡边数。

  纯 host 程序（mmap 读/写 + 显式队列 BFS），无 GPU kernel，可本机编译运行。
  复用 csr_graph.cu 的 readFromGR / writeToGR，零新依赖。

  Usage: ./reorder_gr <input.gr> <output.gr>
*/

#include "csr_graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cerrno>
#include <climits>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <endian.h>
#include <cstring>
#include <algorithm>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Sidecar format: nnodes little-endian uint32 entries, no header.
static bool write_mapping(const std::string &path, const int *mapping, int count) {
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd < 0) { perror(path.c_str()); return false; }
    FILE *stream = fdopen(fd, "wb");
    if (!stream) { close(fd); return false; }
    bool ok = true;
    for (int i = 0; i < count && ok; ++i) {
        uint32_t value = htole32(static_cast<uint32_t>(mapping[i]));
        ok = fwrite(&value, sizeof(value), 1, stream) == 1;
    }
    return fclose(stream) == 0 && ok;
}

static int count_cross_edges(int m, int *rs, int *dst, int cut)
{
    int c = 0;
    for (int u = 0; u < m; u++)
        for (int e = rs[u]; e < rs[u + 1]; e++)
        {
            int v = dst[e];
            if ((u < cut) != (v < cut))
                c++;
        }
    return c;
}

int main(int argc, char *argv[])
{
    if (argc < 3 || argc > 5)
    {
        printf("Usage: ./reorder_gr <input.gr> <output.gr> [BFS-root] [--layer-split]\n");
        return -1;
    }

    char *in_file = argv[1];
    char *out_file = argv[2];
    const double start = now_ms();
    long root = 0;
    bool layer_split = argc == 5;
    if (layer_split && strcmp(argv[4], "--layer-split")) return 2;
    if (argc >= 4) {
        char *end;
        errno = 0;
        root = strtol(argv[3], &end, 10);
        if (errno || end == argv[3] || *end || root < 0 || root > INT_MAX) return 2;
    }

    CSRGraph g;
    float avg_degree, avg_weight;
    if (g.read(in_file, avg_degree, avg_weight))
    {
        printf("Read error!\n");
        return -1;
    }

    int m = g.nnodes;
    int nedges = g.nedges;
    if (m <= 0 || root >= m) { fprintf(stderr, "Invalid graph/root\n"); return 2; }
    const double read_ms = now_ms() - start;
    const double reorder_start = now_ms();
    int cut = m / 2;

    int *rs = g.row_start;
    int *dst = g.edge_dst;
    int *data = g.edge_data;

    int c_before = count_cross_edges(m, rs, dst, cut);

    // BFS 序（root=0，显式队列，防栈溢出）
    int *perm = new int[m];      // perm[new] = old
    int *invperm = new int[m];   // invperm[old] = new，兼任 visited（-1=未访问）
    for (int i = 0; i < m; i++)
        invperm[i] = -1;

    int *q = new int[m];
    int *level = layer_split ? new int[m] : nullptr;
    if (level) { std::fill(level, level + m, -1); level[root] = 0; }
    int head = 0, tail = 0;
    q[tail++] = root;
    invperm[root] = 0;
    perm[0] = root;
    int next = 1;

    while (head < tail)
    {
        int u = q[head++];
        for (int e = rs[u]; e < rs[u + 1]; e++)
        {
            int v = dst[e];
            if (invperm[v] == -1)
            {
                invperm[v] = next;
                if (level) level[v] = level[u] + 1;
                perm[next] = v;
                next++;
                q[tail++] = v;
            }
        }
    }

    // 断连分量按原 id 升序追加
    for (int v = 0; v < m; v++)
        if (invperm[v] == -1)
        {
            invperm[v] = next;
            perm[next] = v;
            next++;
        }

    if (next != m)
    {
        printf("ERROR: BFS order incomplete: next=%d, m=%d\n", next, m);
        return -1;
    }

    if (layer_split) {
        // Divide each BFS level in discovery order, with cumulative rounding
        // keeping exactly floor(n/2) vertices in owner0. Unreachable vertices
        // form one final group. No weighted distances or reference seeding.
        int left = 0, right = cut;
        for (int begin = 0; begin < m;) {
            int end = begin + 1;
            while (end < m && level[perm[end]] == level[perm[begin]]) ++end;
            int take = std::min(cut, (end + 1) / 2) - left;
            for (int i = begin; i < end; ++i) {
                if (i - begin < take) q[left++] = perm[i];
                else q[right++] = perm[i];
            }
            begin = end;
        }
        if (left != cut || right != m) return 2;
        for (int i = 0; i < m; ++i) { perm[i] = q[i]; invperm[q[i]] = i; }
    }

    // 重建新 CSR（new_rs/new_dst/new_data，避免 in-place 边界错误）
    int *new_rs = new int[m + 1];
    int *new_dst = new int[nedges];
    int *new_data = new int[nedges];

    new_rs[0] = 0;
    for (int ni = 0; ni < m; ni++)
    {
        int old = perm[ni];
        new_rs[ni + 1] = new_rs[ni] + (rs[old + 1] - rs[old]);
    }

    int pos = 0;
    for (int ni = 0; ni < m; ni++)
    {
        int old = perm[ni];
        for (int e = rs[old]; e < rs[old + 1]; e++)
        {
            new_dst[pos] = invperm[dst[e]];
            new_data[pos] = data[e];
            pos++;
        }
    }

    // 校验（可靠性兜底）
    if (pos != nedges)
    {
        printf("ERROR: edge count mismatch: pos=%d, nedges=%d\n", pos, nedges);
        return -1;
    }
    if (new_rs[m] != nedges)
    {
        printf("ERROR: row_start convergence: new_rs[m]=%d, nedges=%d\n", new_rs[m], nedges);
        return -1;
    }
    for (int e = 0; e < nedges; e++)
        if (new_dst[e] < 0 || new_dst[e] >= m)
        {
            printf("ERROR: dst domain invalid: new_dst[%d]=%d\n", e, new_dst[e]);
            return -1;
        }
    for (int ni = 0; ni < m; ni++)
        if (new_rs[ni + 1] - new_rs[ni] != rs[perm[ni] + 1] - rs[perm[ni]])
        {
            printf("ERROR: degree mismatch at new vertex %d\n", ni);
            return -1;
        }

    int c_after = count_cross_edges(m, new_rs, new_dst, cut);
    const double reorder_ms = now_ms() - reorder_start;
    const double output_start = now_ms();
    // Reserve all output names; never overwrite an existing graph or mapping.
    // An I/O failure leaves explicit partial artifacts, which must not be benchmarked.
    const std::string perm_path = std::string(out_file) + ".perm.u32";
    const std::string inv_path = std::string(out_file) + ".invperm.u32";
    if (access(out_file, F_OK) == 0 || access(perm_path.c_str(), F_OK) == 0 ||
        access(inv_path.c_str(), F_OK) == 0) {
        fprintf(stderr, "Output already exists; use a new path\n"); return 2;
    }
    int out_fd = open(out_file, O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (out_fd < 0) { perror(out_file); return 2; }
    close(out_fd);
    if (!write_mapping(perm_path, perm, m) || !write_mapping(inv_path, invperm, m)) return 2;

    if (writeToGR(m, nedges, new_rs, new_dst, new_data, out_file))
    {
        printf("Write error!\n");
        return -1;
    }

    double pct_before = nedges ? 100.0 * (double)c_before / (double)nedges : 0;
    double pct_after = nedges ? 100.0 * (double)c_after / (double)nedges : 0;
    printf("REORDER layout=%s root_old=%ld root_new=0 vertices=%d edges=%d read_ms=%.6f "
           "reorder_ms=%.6f output_ms=%.6f total_ms=%.6f\n", layer_split ? "bfs_layer_split" : "bfs", root, m, nedges,
           read_ms, reorder_ms, now_ms() - output_start, now_ms() - start);
    printf("cross_edges before: %d (%.2f%%)\n", c_before, pct_before);
    printf("cross_edges after : %d (%.2f%%)\n", c_after, pct_after);
    if (c_after > 0)
        printf("reduction: %.1fx\n", (double)c_before / (double)c_after);
    else
        printf("reduction: inf\n");

    delete[] perm;
    delete[] invperm;
    delete[] q;
    delete[] level;
    delete[] new_rs;
    delete[] new_dst;
    delete[] new_data;

    return 0;
}
