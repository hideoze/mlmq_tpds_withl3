#include <cassert>

#ifndef L3_WINDOW_MIN_CYCLES
#define L3_WINDOW_MIN_CYCLES 25000ull
#endif
#ifndef L3_WINDOW_MAX_CYCLES
#define L3_WINDOW_MAX_CYCLES L3_WINDOW_MIN_CYCLES
#endif

#include "../../SSSP/l3/l3_window.h"

int main() {
    constexpr unsigned long long minimum = L3_WINDOW_MIN_CYCLES;
    constexpr unsigned long long maximum = L3_WINDOW_MAX_CYCLES;
    static_assert(minimum > 0);
    static_assert(maximum >= minimum);

    l3_window_state state;
    assert(state.budget == minimum);
    assert(state.decision(100, true, maximum) == l3_window_state::WAIT);
    assert(state.decision(100 + minimum - 1, true, maximum) == l3_window_state::WAIT);
    assert(state.decision(100 + minimum, true, maximum) == l3_window_state::SCAN);

    l3_window_state fixed;
    assert(fixed.decision(100, false, maximum) == l3_window_state::WAIT);
    assert(fixed.decision(100 + maximum - 1, false, maximum) ==
           l3_window_state::WAIT);
    assert(fixed.decision(100 + maximum, false, maximum) ==
           l3_window_state::SCAN);

    state.scanned(100 + minimum, 0, 0, true, maximum);
    const auto backed_off = minimum > maximum / 2 ? maximum : minimum * 2;
    assert(state.budget == backed_off);
    if (maximum > minimum) {
        assert(state.decision(100 + 2 * minimum, true, maximum) == l3_window_state::PROBE);
    }

    l3_window_state drain;
    assert(!drain.allow(500, true, true, maximum));
    assert(!drain.allow(500 + minimum - 1, true, true, maximum));
    assert(drain.allow(500 + minimum, true, true, maximum));

    l3_window_state feedback;
    feedback.budget = maximum;
    feedback.effective_feedback(8, 8, maximum);
    assert(feedback.budget >= minimum && feedback.budget <= maximum);
    feedback.effective_feedback(8, 0, maximum);
    assert(feedback.budget >= minimum && feedback.budget <= maximum);
    return 0;
}
