
### Nodes and Partition Mapping

| Node                    | SLURM Partition              | GPU           | CC    | Notes                                                                                         |
| ----------------------- | ---------------------------- | ------------- | ----- | --------------------------------------------------------------------------------------------- |
| ada-709 (local machine) | Non-compute node (Gres=null) | 2×RTX 3080Ti | sm_86 | ⚠️ Editing/compilation only; driver unavailable; running GPUs on this machine is prohibited |
| ada-4090                | 4090                         | 4×RTX 4090   | sm_89 | Driver 580.126 ✅                                                                             |
| ada-A100                | a100                         | 2×A100 80GB  | sm_80 | Driver 570.133 ✅, NVLink P2P                                                                 |

- **The Makefile's gencode (sm_xx) must match the GPU architecture of the target partition**; verify this before running on a different partition.

### How to Run (Important)

- **All GPU jobs must be submitted through Slurm** (`sbatch`/`srun`). The local ada-709 is only for code editing and compilation.
- A driver version mismatch was previously confirmed on this machine; this is an environment issue, not an algorithm issue. Do not debug on this machine or bypass driver restrictions to run GPUs. The driver versions in the table are historical records and do not represent live status.

### GPU Usage Checks

- First check resources with `squeue`. After obtaining the required Slurm allocation, use `nvidia-smi` on the **target compute node** to check the allocated GPUs. You can complete both the check and the experiment within the same job; there is no need to submit a separate check job.
- **Only select GPUs that are not in use by anyone else for experiments.**
- When using a partition for the first time, also run `srun ... nvidia-smi topo -m` to confirm the GPU topology.

### Slurm Usage Guidelines

- When submitting jobs, you must specify reasonable memory via `--mem` (e.g., `--mem=16G`); otherwise, Slurm requests all node memory by default, which may cause jobs to queue.
- Only clean up Dingzhong jobs or processes that are confirmed to belong to this task and are indeed leftover; do not use usernames as the basis for bulk cleanup. **It is strictly prohibited to clean up other users' jobs or processes.**
- If resources are insufficient, remind the user; do not repeatedly retry and cause queuing.
