#pragma once

// All lanes participate; all receive the same answer. Hints are deliberately
// absent: an empty result must inspect every possible authoritative word. A nonempty
// tile permits immediate whole-warp rejection instead of divergent long tails.
__device__ __forceinline__ bool l3_marks_empty_warp(
    const unsigned *mark, int words, int lane)
{
    if (!mark) return true;
#if (L3_BOUNDARY_INDEX == true)
    const int count=l3_scan_word_count(words);
#else
    const int count=words;
#endif
    for (int base=0; base<count; base+=32) {
        int index=base+lane;
#if (L3_BOUNDARY_INDEX == true)
        int word=index<count ? l3_scan_word(index) : words;
#else
        int word=index;
#endif
        bool live=index<count && *((volatile const unsigned *)(mark+word))!=0;
        if (__any_sync(0xffffffffu,live)) return false;
    }
    return true;
}
