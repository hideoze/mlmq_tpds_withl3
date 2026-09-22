#pragma once
#include "l3_feedback.h"
#include "l3_window.h"

#if (L3_RX_FEEDBACK_MODE > 0)
// TX lane zero calls once after each successful publication, including a
// retained retry. The record was copied under the slot's WRITING ownership.
__device__ __forceinline__ void l3_apply_rx_feedback(
    l3_window_state &window, const l3_feedback_record &record,
    int tx_epoch, int owner, int receiver)
{
    if (!record.valid(tx_epoch - BULK_INBOX_SLOTS)) return;
    const auto before = window.budget;
#if (L3_RX_FEEDBACK_MODE == 2)
    window.effective_feedback(record.count, record.improved, L3_WINDOW_MAX_CYCLES);
#endif
#if (L3_DIAGNOSTICS == true)
    ++g_l3_diagnostics.feedback_batches;
    g_l3_diagnostics.feedback_count += record.count;
    g_l3_diagnostics.feedback_improved += record.improved;
    g_l3_diagnostics.feedback_changes += before != window.budget;
#endif
#if (L3_RX_FEEDBACK_TRACE == true)
    l3_trace_feedback(1, owner, receiver, tx_epoch, record, before, window.budget, L3_RX_FEEDBACK_MODE);
#endif
}
#endif
