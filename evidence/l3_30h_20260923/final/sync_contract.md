# BNUM=8 re-freeze synchronization contract

This audit is scoped to the current re-freeze BULK configuration built by
`scripts/multigpu/build_l3_paper_config.sh`: W512, direct RX, retained TX,
worker recovery, termination ACK scanning, boundary index, L2 final counts,
guarded 32-bit DQ counters, fixed 25,000-cycle scheduling window, and no
compact-candidate or diagnostic hot-path variant.  Dormant `SEED_BARRIER`,
tile-loan, owner-commit, and other
experimental branches are not covered by the performance claim.

This contract describes the current BNUM=8 re-freeze candidate, not an
already accepted formal result.  Both the dual build and the independent
no-L3 single build use explicit canonical definitions `BNUM=8`,
`BUCKET_MAX=4`, and `l2_batch_size=8`; the formal runner checks all three
definitions.  Its L2 budget is 2,147,483,647 bytes (`INT_MAX`), of which
2,147,483,640 bytes hold 268,435,455 eight-byte records; each bucket exposes
33,553,920 records and the total usable bucket region is 268,431,360 records.
A new clean commit still has to pass the full Job A and independent Job B
workflows.

The synchronization implementation is the complete `567e47f` repair, with
the later window parameterization and effective-configuration logging layered
on top.  The window changes only when work is scheduled; they do not remove a
publication, ACK, recovery, or termination condition.

| Handoff | Publisher | Observer | Scope and order in the frozen source |
|---|---|---|---|
| candidate value -> mark | `l3_record_candidate` in `SSSP/l3/l3_candidate.cuh` | cooperative/full recovery collectors in `l3_collect.cuh`, `l3_bulk.cuh`, and `SSSP/sssp_run.cu` | candidate `atomicMin`, then device-release mark publication; collectors use device-acquire load/claim. `remote_mark` remains authoritative; hints remain advisory. |
| payload/count/epoch -> READY | `bulk_pack_publish_warp` / `bulk_publish_l3_batch` in `l3_bulk.cuh` | `bulk_inbox_claim_read` plus `bulk_inbox_confirm_read_lane` in `l3_transport.cuh` and RX consumers | payload stores precede system-release READY; claim is system acquire/acq-rel, and every payload-reading lane performs an acquire observation of READING. |
| RX commit -> ACK/DONE | `l3_receive_to_l2` in `l3_receive.cuh` | sender slot reuse in `bulk_inbox_try_acquire_write` | distance/dirty/L2 publication completes before system-release ACK and DONE; sender performs system-acquire observations and also verifies generation. |
| DQ payload -> publication -> read | `l2_delta_queue::write` in `core/cu_delta_queue/cu_delta_queue.cuh` | `manager_run`, then queue readers | producing lanes fence payload before release publication to `block_write_done`; manager acquires completion and release-publishes `read_pos`; readers acquire `read_pos`. |
| publish failure / backpressure | retained journal and requeue paths in `SSSP/sssp_run.cu` and `l3_bulk.cuh` | later TX/recovery attempts | an unaccepted batch remains retained or is release-republished into the authoritative candidate marks; no candidate is discarded merely because a slot is full. |
| worker recovery and termination | worker request/done, work/inject/TX ACKs, peer state | manager full predicate and final exit | device/system release-acquire operations order request, ACK, exact mark scan, latest transport epoch, recovery, peer READY, manager end, and global exit. ACK geometry is checked against the active W512 launch. |
| authoritative DQ write reservation | each per-bucket `write_reserve` RMW | the same queue operation and final host audit | the formal guard is a cumulative no-physical-wrap gate, not merely a signed-integer overflow check. `write_reserve` is cumulative while payload addresses use modulo `total_size`; because the current queue has no proven contiguous retire frontier or generation protocol for safe physical-slot reuse, every successful per-bucket reservation must remain within `total_size`. A violating reservation sets the sticky flag and executes a device trap before the returned slot can be used. `overflow_detected=0` and `no_wrap=1` therefore prove that no physical reuse occurred; they do not prove that arbitrary ring reuse would be safe. Downstream read/completion counters remain bounded by unique successful reservations, and the independent single build uses the same shared-core guard. |
| post-query L2 completion | synchronized host-side final counter read | formal runner | `mlmq_benchmark::finish` is a post-timer host barrier: both GPU threads complete `cudaDeviceSynchronize` before either reads final counters. Every bucket must then be drained; guarded writes, reads, and completions must agree; the sticky overflow flag must be zero. `L2_FINAL` is cross-checked against the process's `L2_CAPACITY`, rather than inferring no-wrap from conservation alone. |

Early `l3_busy`, hint, and cached published snapshots are scheduling hints.
The final empty/termination decision re-reads authoritative marks, transport
epochs, ACKs, recovery state, and peer state after the required acquire
operations.  This audit therefore does not claim that every legacy advisory
field is itself race-free.

Changing BNUM from 16 to 8 is not a storage-only repair.  It changes manager
warps from 16 to 8, shared metadata, the visible delta-bucket horizon,
far-distance clamping, and work order.  The 4,095 records outside the rounded
bucket regions are not advertised as usable capacity.  Consequently all T1
and T2 measurements, extra sources, and graph regressions must be collected
again; no BNUM=16 timing may be reused for the BNUM=8 candidate.

Static inspection makes the active default path acceptable for runtime
validation; it is not a substitute for that validation.  Formal acceptance
still requires the exact clean-SHA build to pass candidate interleaving,
full-slot retry, delayed ACK, RX-before-ACK, DQ publication, sequential reset,
capacity/no-wrap checks, per-query oracle checks, and the final paired run.
The final commit SHA, builder/source hashes, and the complete enabled/disabled
macro vector are filled from the clean pair-build evidence after the freeze;
the current document deliberately does not predict that SHA.

The BNUM=16 clean-SHA attempt `7cd563c`, Slurm job 38238, failed with CUDA
719 during the second query of the round-1 dual process after a correct
warmup.  It is retained as invalid failure evidence and is not part of any
accepted performance aggregate.  The active DQ guard is the high-confidence
cause based on its compiled trap path and the observed bucket pressure, but no
trap program counter was recovered, so this is not stated as instruction-level
proof.
