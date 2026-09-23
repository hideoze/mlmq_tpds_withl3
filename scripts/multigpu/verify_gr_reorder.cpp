// Verify that a Galois v1 GR file is exactly a vertex relabeling of another.
// This is intentionally host-only and independent of the solver's GR reader.
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

uint64_t checked_add(uint64_t left, uint64_t right, const char *what) {
    if (right > std::numeric_limits<uint64_t>::max() - left) {
        fail(std::string("size overflow: ") + what);
    }
    return left + right;
}

uint64_t checked_mul(uint64_t left, uint64_t right, const char *what) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
        fail(std::string("size overflow: ") + what);
    }
    return left * right;
}

uint32_t load_le32(const unsigned char *bytes) {
    return uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) |
           (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
}

uint64_t load_le64(const unsigned char *bytes) {
    uint64_t value = 0;
    for (unsigned shift = 0; shift != 64; shift += 8) {
        value |= uint64_t(bytes[shift / 8]) << shift;
    }
    return value;
}

class MappedFile {
  public:
    explicit MappedFile(const char *path) : path_(path) {
        const int descriptor = open(path, O_RDONLY | O_CLOEXEC);
        if (descriptor < 0) {
            fail(path_ + ": open: " + std::strerror(errno));
        }

        struct stat status {};
        if (fstat(descriptor, &status) != 0) {
            const int saved_errno = errno;
            close(descriptor);
            fail(path_ + ": fstat: " + std::strerror(saved_errno));
        }
        if (status.st_size < 0 ||
            uintmax_t(status.st_size) > uintmax_t(std::numeric_limits<size_t>::max())) {
            close(descriptor);
            fail(path_ + ": file is too large to mmap");
        }

        const size_t length = size_t(status.st_size);
        void *mapping = nullptr;
        if (length != 0) {
            mapping = mmap(nullptr, length, PROT_READ, MAP_PRIVATE, descriptor, 0);
            if (mapping == MAP_FAILED) {
                const int saved_errno = errno;
                close(descriptor);
                fail(path_ + ": mmap: " + std::strerror(saved_errno));
            }
        }
        fd_ = descriptor;
        size_ = length;
        data_ = static_cast<const unsigned char *>(mapping);
    }

    ~MappedFile() {
        if (data_ != nullptr) munmap(const_cast<unsigned char *>(data_), size_);
        if (fd_ >= 0) close(fd_);
    }

    MappedFile(const MappedFile &) = delete;
    MappedFile &operator=(const MappedFile &) = delete;

    const unsigned char *range(uint64_t offset, uint64_t length) const {
        if (offset > uint64_t(size_) || length > uint64_t(size_) - offset) {
            fail(path_ + ": truncated range offset=" + std::to_string(offset) +
                 " bytes=" + std::to_string(length));
        }
        if (data_ == nullptr) return nullptr;
        return data_ + size_t(offset);
    }

    size_t size() const { return size_; }
    const std::string &path() const { return path_; }

  private:
    std::string path_;
    int fd_ = -1;
    size_t size_ = 0;
    const unsigned char *data_ = nullptr;
};

struct Graph {
    explicit Graph(const char *path) : file(path) {
        if (file.size() < 32) fail(file.path() + ": short GR header");
        const unsigned char *header = file.range(0, 32);
        version = load_le64(header);
        edge_type_size = load_le64(header + 8);
        vertices = load_le64(header + 16);
        edges = load_le64(header + 24);

        if (version != 1) {
            fail(file.path() + ": unsupported GR version " + std::to_string(version));
        }
        if (vertices > uint64_t(std::numeric_limits<uint32_t>::max())) {
            fail(file.path() + ": vertex count exceeds uint32 mapping domain");
        }

        const uint64_t row_bytes = checked_mul(vertices, 8, "GR rows");
        const uint64_t destination_bytes = checked_mul(edges, 4, "GR destinations");
        row_offset = 32;
        destination_offset = checked_add(row_offset, row_bytes, "destination offset");
        destination_padding = (edges & 1) ? 4 : 0;
        payload_offset = checked_add(
            checked_add(destination_offset, destination_bytes, "destination end"),
            destination_padding, "payload alignment");
        payload_bytes = checked_mul(edges, edge_type_size, "edge payload");
        expected_bytes = checked_add(payload_offset, payload_bytes, "GR payload end");
        file.range(0, expected_bytes);

        rows = file.range(row_offset, row_bytes);
        destinations = file.range(destination_offset, destination_bytes);
        payload = file.range(payload_offset, payload_bytes);
        trailing_bytes = uint64_t(file.size()) - expected_bytes;

        if (destination_padding != 0) {
            const unsigned char *padding = file.range(
                destination_offset + destination_bytes, destination_padding);
            for (uint64_t index = 0; index < destination_padding; ++index) {
                if (padding[index] != 0) {
                    fail(file.path() + ": nonzero destination padding");
                }
            }
        }
        const unsigned char *trailing = file.range(expected_bytes, trailing_bytes);
        for (uint64_t index = 0; index < trailing_bytes; ++index) {
            if (trailing[index] != 0) {
                fail(file.path() + ": nonzero trailing byte at offset " +
                     std::to_string(expected_bytes + index));
            }
        }

        uint64_t previous = 0;
        for (uint64_t vertex = 0; vertex < vertices; ++vertex) {
            const uint64_t end = row_end(vertex);
            if (end < previous || end > edges) {
                fail(file.path() + ": invalid row end at vertex " +
                     std::to_string(vertex));
            }
            previous = end;
        }
        if (previous != edges) {
            fail(file.path() + ": final row end does not equal edge count");
        }
        for (uint64_t edge = 0; edge < edges; ++edge) {
            if (destination(edge) >= vertices) {
                fail(file.path() + ": destination out of range at edge " +
                     std::to_string(edge));
            }
        }
    }

    uint64_t row_begin(uint64_t vertex) const {
        return vertex == 0 ? 0 : row_end(vertex - 1);
    }
    uint64_t row_end(uint64_t vertex) const {
        return load_le64(rows + vertex * 8);
    }
    uint32_t destination(uint64_t edge) const {
        return load_le32(destinations + edge * 4);
    }
    const unsigned char *edge_payload(uint64_t edge) const {
        return payload + edge * edge_type_size;
    }

    MappedFile file;
    uint64_t version = 0;
    uint64_t edge_type_size = 0;
    uint64_t vertices = 0;
    uint64_t edges = 0;
    uint64_t row_offset = 0;
    uint64_t destination_offset = 0;
    uint64_t destination_padding = 0;
    uint64_t payload_offset = 0;
    uint64_t payload_bytes = 0;
    uint64_t expected_bytes = 0;
    uint64_t trailing_bytes = 0;
    const unsigned char *rows = nullptr;
    const unsigned char *destinations = nullptr;
    const unsigned char *payload = nullptr;
};

struct Mapping {
    Mapping(const char *path, uint64_t entries) : file(path), count(entries) {
        const uint64_t expected = checked_mul(entries, 4, "mapping");
        if (uint64_t(file.size()) != expected) {
            fail(file.path() + ": expected " + std::to_string(expected) +
                 " bytes, got " + std::to_string(file.size()));
        }
        values = file.range(0, expected);
    }

    uint32_t operator[](uint64_t index) const {
        return load_le32(values + index * 4);
    }

    MappedFile file;
    uint64_t count;
    const unsigned char *values = nullptr;
};

struct Oracle {
    Oracle(const char *path, uint64_t entries) : file(path), count(entries) {
        const uint64_t expected = checked_mul(entries, 4, "oracle");
        if (uint64_t(file.size()) != expected) {
            fail(file.path() + ": expected " + std::to_string(expected) +
                 " bytes, got " + std::to_string(file.size()));
        }
        values = file.range(0, expected);
    }

    uint32_t bits(uint64_t index) const {
        return load_le32(values + index * 4);
    }

    MappedFile file;
    uint64_t count;
    const unsigned char *values = nullptr;
};

}  // namespace

int main(int argc, char **argv) try {
    if (argc != 5 && argc != 7) {
        std::fprintf(stderr,
                     "usage: %s ORIGINAL REORDERED PERM_U32 INVPERM_U32 "
                     "[ORIGINAL_ORACLE_I32 REORDERED_ORACLE_I32]\n",
                     argv[0]);
        return 2;
    }

    const Graph original(argv[1]);
    const Graph reordered(argv[2]);
    if (original.version != reordered.version ||
        original.edge_type_size != reordered.edge_type_size ||
        original.vertices != reordered.vertices || original.edges != reordered.edges) {
        fail("GR header mismatch between original and reordered graphs");
    }

    const uint64_t vertices = original.vertices;
    const uint64_t edges = original.edges;
    const Mapping perm(argv[3], vertices);       // perm[new] = old
    const Mapping invperm(argv[4], vertices);    // invperm[old] = new
    std::vector<unsigned char> seen_old(size_t(vertices), 0);
    std::vector<unsigned char> seen_new(size_t(vertices), 0);

    for (uint64_t new_vertex = 0; new_vertex < vertices; ++new_vertex) {
        const uint32_t old_vertex = perm[new_vertex];
        if (old_vertex >= vertices) {
            fail("perm out of range at new vertex " + std::to_string(new_vertex));
        }
        if (seen_old[old_vertex] != 0) {
            fail("perm duplicate old vertex " + std::to_string(old_vertex));
        }
        seen_old[old_vertex] = 1;
        if (invperm[old_vertex] != new_vertex) {
            fail("invperm[perm[new]] != new at new vertex " +
                 std::to_string(new_vertex));
        }
    }
    for (uint64_t old_vertex = 0; old_vertex < vertices; ++old_vertex) {
        const uint32_t new_vertex = invperm[old_vertex];
        if (new_vertex >= vertices) {
            fail("invperm out of range at old vertex " + std::to_string(old_vertex));
        }
        if (seen_new[new_vertex] != 0) {
            fail("invperm duplicate new vertex " + std::to_string(new_vertex));
        }
        seen_new[new_vertex] = 1;
        if (perm[new_vertex] != old_vertex) {
            fail("perm[invperm[old]] != old at old vertex " +
                 std::to_string(old_vertex));
        }
    }

    uint64_t destination_checks = 0;
    uint64_t payload_checks = 0;
    for (uint64_t new_vertex = 0; new_vertex < vertices; ++new_vertex) {
        const uint32_t old_vertex = perm[new_vertex];
        const uint64_t old_begin = original.row_begin(old_vertex);
        const uint64_t old_end = original.row_end(old_vertex);
        const uint64_t new_begin = reordered.row_begin(new_vertex);
        const uint64_t new_end = reordered.row_end(new_vertex);
        if (old_end - old_begin != new_end - new_begin) {
            fail("degree mismatch at new vertex " + std::to_string(new_vertex) +
                 " old vertex " + std::to_string(old_vertex));
        }

        for (uint64_t offset = 0; offset < old_end - old_begin; ++offset) {
            const uint64_t old_edge = old_begin + offset;
            const uint64_t new_edge = new_begin + offset;
            const uint32_t old_destination = original.destination(old_edge);
            const uint32_t expected_new_destination = invperm[old_destination];
            const uint32_t actual_new_destination = reordered.destination(new_edge);
            if (actual_new_destination != expected_new_destination) {
                fail("destination mismatch at new vertex " +
                     std::to_string(new_vertex) + " row offset " +
                     std::to_string(offset));
            }
            ++destination_checks;

            if (original.edge_type_size != 0) {
                if (std::memcmp(original.edge_payload(old_edge),
                                reordered.edge_payload(new_edge),
                                size_t(original.edge_type_size)) != 0) {
                    fail("edge payload mismatch at new vertex " +
                         std::to_string(new_vertex) + " row offset " +
                         std::to_string(offset));
                }
                ++payload_checks;
            }
        }
    }
    if (destination_checks != edges ||
        (original.edge_type_size != 0 && payload_checks != edges)) {
        fail("internal edge-check count mismatch");
    }

    uint64_t oracle_checks = 0;
    if (argc == 7) {
        const Oracle original_oracle(argv[5], vertices);
        const Oracle reordered_oracle(argv[6], vertices);
        for (uint64_t new_vertex = 0; new_vertex < vertices; ++new_vertex) {
            const uint32_t old_vertex = perm[new_vertex];
            if (reordered_oracle.bits(new_vertex) != original_oracle.bits(old_vertex)) {
                fail("oracle mismatch at new vertex " + std::to_string(new_vertex) +
                     " old vertex " + std::to_string(old_vertex));
            }
            ++oracle_checks;
        }
    }

    std::printf(
        "REORDER_VERIFY_PASS vertices=%llu edges=%llu header_fields=4 "
        "perm_checks=%llu invperm_checks=%llu degree_checks=%llu "
        "destination_checks=%llu payload_checks=%llu payload_bytes_per_edge=%llu "
        "oracle_checks=%llu original_trailing_bytes=%llu reordered_trailing_bytes=%llu\n",
        static_cast<unsigned long long>(vertices),
        static_cast<unsigned long long>(edges),
        static_cast<unsigned long long>(vertices),
        static_cast<unsigned long long>(vertices),
        static_cast<unsigned long long>(vertices),
        static_cast<unsigned long long>(destination_checks),
        static_cast<unsigned long long>(payload_checks),
        static_cast<unsigned long long>(original.edge_type_size),
        static_cast<unsigned long long>(oracle_checks),
        static_cast<unsigned long long>(original.trailing_bytes),
        static_cast<unsigned long long>(reordered.trailing_bytes));
    return 0;
} catch (const std::exception &error) {
    std::fprintf(stderr, "REORDER_VERIFY_FAIL message=%s\n", error.what());
    return 1;
}
