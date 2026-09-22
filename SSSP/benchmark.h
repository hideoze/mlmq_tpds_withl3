#pragma once
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <cerrno>

inline int bench_env_int(const char *name, int fallback, int minimum, int maximum) {
    const char *value = std::getenv(name);
    if (!value) return fallback;
    char *end = nullptr;
    errno = 0;
    long parsed = std::strtol(value, &end, 10);
    if (errno || end == value || *end || parsed < minimum || parsed > maximum) {
        fprintf(stderr, "Invalid %s=%s; expected [%d,%d]\n", name, value, minimum, maximum);
        std::exit(2);
    }
    return static_cast<int>(parsed);
}

inline double mlmq_bench_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Host-only, single-query barrier. Configure before launching worker threads.
// No distance data or device protocol state participates in this clock.
struct mlmq_benchmark {
    bool enabled = false;
    int participants = 0, ready = 0, finished = 0;
    double start = 0, end = 0;
    std::mutex mutex;
    std::condition_variable cv;

    void begin() {
        if (!enabled) return;
        std::unique_lock<std::mutex> lock(mutex);
        if (++ready == participants) {
            start = mlmq_bench_ms();
            cv.notify_all();
        } else {
            cv.wait(lock, [this] { return ready == participants; });
        }
    }
    void finish() {
        if (!enabled) return;
        std::lock_guard<std::mutex> lock(mutex);
        if (++finished == participants) end = mlmq_bench_ms();
    }
    void require(bool ok, const char *message) const {
        if (enabled && !ok) {
            fprintf(stderr, "MLMQ BENCH failure: %s\n", message);
            // Exit whole process: a peer may already be waiting in its kernel.
            std::exit(2);
        }
    }
};
extern mlmq_benchmark g_benchmark;
