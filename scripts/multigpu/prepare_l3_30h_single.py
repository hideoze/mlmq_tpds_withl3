#!/usr/bin/env python3
"""Add an auditable work-block control to a copied no-L3 single source tree."""

import argparse
from pathlib import Path


OLD = """        int work_block_num = dev_prop.multiProcessorCount - 1;
        // int work_block_num = 1;
"""

NEW = """        int work_block_num = getenv("MLMQ_WORK_BLOCKS")
            ? atoi(getenv("MLMQ_WORK_BLOCKS"))
            : dev_prop.multiProcessorCount - 1;
        if (work_block_num < 1 || work_block_num > dev_prop.multiProcessorCount - 1) {
            fprintf(stderr, "NO_L3 invalid MLMQ_WORK_BLOCKS=%d allowed=[1,%d]\\n",
                    work_block_num, dev_prop.multiProcessorCount - 1);
            std::exit(2);
        }
        printf("NO_L3_LAUNCH work_blocks=%d delta=%d repeat=%d\\n",
               work_block_num, setup.s_l2_delta, repeat);
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    args = parser.parse_args()
    path = args.source_root / "SSSP/sssp_run.cu"
    raw = path.read_text()
    if raw.count(OLD) != 1:
        parser.error("expected exactly one historical work_block_num launch site")
    if "NO_L3_LAUNCH work_blocks=" in raw:
        parser.error("source already contains the 30-hour single adapter")
    path.write_text(raw.replace(OLD, NEW))
    print(f"PREPARE_L3_30H_SINGLE PASS path={path}")


if __name__ == "__main__":
    main()
