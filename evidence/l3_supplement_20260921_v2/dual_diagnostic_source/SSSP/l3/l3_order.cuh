#pragma once

// Warp-uniform in-place sorting of the sealed journal. Padding only touches
// unused capacity. No candidate is removed and no epoch is advanced here.
__device__ __forceinline__ void l3_order_batch(
    int *ids, int *values, int count, int lane)
{
    bool descending = false;
    for (int i = lane + 1; i < count; i += 32)
        descending |= values[i] < values[i - 1];
    if (!__any_sync(0xffffffffu, descending)) return;
    int length = 1;
    while (length < count) length <<= 1;
    for (int i = count + lane; i < length; i += 32) {
        ids[i] = 0;
        values[i] = 2147483647;
    }
    __syncwarp();
    for (int width = 2; width <= length; width <<= 1) {
        for (int stride = width >> 1; stride; stride >>= 1) {
            for (int i = lane; i < length; i += 32) {
                int other = i ^ stride;
                if (other > i) {
                    bool ascending = (i & width) == 0;
                    int a = values[i], b = values[other];
                    if ((ascending && a > b) || (!ascending && a < b)) {
                        values[i] = b; values[other] = a;
                        int id = ids[i]; ids[i] = ids[other]; ids[other] = id;
                    }
                }
            }
            __syncwarp();
        }
    }
}
