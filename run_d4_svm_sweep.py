"""
D4 SVM Sweep — Group 52, COMP3850
Author: Jeslin

Matches the team's fixed parameter set from Amora's spreadsheet:
  BF_length     = 1000
  BF_num_hash   = 10
  BF_q_gram     = 2
  min_sim_val   = 0.6        ← matches Amora's sheet (NOT 0.5 from notebook)
  link_attrs    = [1, 2, 3, 4]
  block_attrs   = [2, 4]     ← fixed per Amora's sheet
  ent_index     = 0
  encoding      = 'cbf'      ← Counting Bloom Filter (current notebook)
  epsilon       = varies per row (1, 2, 3, 4, 5)

Sweep: 6 datasets × 5 epsilon values = 30 rows
(If team only needs ε ∈ {1, 2, 3} like Amora's sheet, set EPSILONS = [1, 2, 3]
 to match her 18 rows exactly.)

Usage:
  Place in the same folder as PPRL.py and BF.py
  Run: python run_d4_svm_sweep.py
"""

import csv
import math
import time
from PPRL import Link


# ---------- preset parameters (matches Amora's sheet) ----------
BF_LENGTH = 1000
BF_NUM_HASH = 10
BF_Q_GRAM = 2
MIN_SIM_VAL = 0.6
LINK_ATTRS = [1, 2, 3, 4]
BLOCK_ATTRS = [2, 4]
ENT_INDEX = 0
ENCODING = 'cbf'

# ---------- epsilon values to run ----------
# Set to [1, 2, 3] to match Amora's 18-row sheet exactly.
# Set to [1, 2, 3, 4, 5] for the full sweep mentioned in the Slack message.
EPSILONS = [1, 2, 3, 4, 5]

# ---------- datasets (order matches Amora's RunID order) ----------
DATASETS = [
    ('100_corr_25',  'Datasets/Alice_numrec_100_corr_25.csv',  'Datasets/Bob_numrec_100_corr_25.csv'),
    ('100_corr_50',  'Datasets/Alice_numrec_100_corr_50.csv',  'Datasets/Bob_numrec_100_corr_50.csv'),
    ('100_non_corr', 'Datasets/Alice_numrec_100_non_corr.csv', 'Datasets/Bob_numrec_100_non_corr.csv'),
    ('500_corr_25',  'Datasets/Alice_numrec_500_corr_25.csv',  'Datasets/Bob_numrec_500_corr_25.csv'),
    ('500_corr_50',  'Datasets/Alice_numrec_500_corr_50.csv',  'Datasets/Bob_numrec_500_corr_50.csv'),
    ('500_non_corr', 'Datasets/Alice_numrec_500_non_corr.csv', 'Datasets/Bob_numrec_500_non_corr.csv'),
]


def run_one(alice_path, bob_path, epsilon):
    """One experiment — returns (precision, recall, f1, runtime_s)."""
    t0 = time.time()

    link = Link(
        length=BF_LENGTH,
        num_hash_func=BF_NUM_HASH,
        q=BF_Q_GRAM,
        min_sim_val=MIN_SIM_VAL,
        use_attr_index=LINK_ATTRS,
        blk_attr_index=BLOCK_ATTRS,
        ent_id=ENT_INDEX,
        epsilon=epsilon,
    )

    db1 = link.read_database(alice_path)
    db2 = link.read_database(bob_path)

    blk1 = link.build_BI(db1)
    blk2 = link.build_BI(db2)

    bf1, _ = link.data_encode(db1, encoding=ENCODING)
    bf2, _ = link.data_encode(db2, encoding=ENCODING)

    pbf1 = link.add_DP_noise(bf1)
    pbf2 = link.add_DP_noise(bf2)

    matches, precision, recall, f1 = link.match_svm(blk1, blk2, pbf1, pbf2, db1, db2)
    # metrics computed inside match_svm now

    return precision, recall, f1, time.time() - t0


def main():
    out_path = 'd4_svm_sweep_results.csv'
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        # header matches Amora's column order
        w.writerow(['RunID', 'Dataset', 'epsilon', 'Precision', 'Recall', 'F1', 'Runtime_s'])

        run_id = 0
        total = len(DATASETS) * len(EPSILONS)

        # Order: dataset outer, epsilon inner — matches Amora (100_corr_25 ε=1,2,3 then 100_corr_50 ε=1,2,3 ...)
        for ds_name, alice, bob in DATASETS:
            for eps in EPSILONS:
                run_id += 1
                print(f"\n[{run_id}/{total}] {ds_name}  eps={eps}")
                try:
                    p, r, f1, rt = run_one(alice, bob, eps)
                    w.writerow([run_id, ds_name, eps,
                                round(p, 6), round(r, 6), round(f1, 6),
                                round(rt, 2)])
                    f.flush()
                    print(f"   P={p:.4f}  R={r:.4f}  F1={f1:.4f}  ({rt:.1f}s)")
                except Exception as e:
                    print(f"   ERROR: {e}")
                    w.writerow([run_id, ds_name, eps, 'ERR', 'ERR', 'ERR', 'ERR'])
                    f.flush()

    print(f"\nDone. Results saved to: {out_path}")


if __name__ == '__main__':
    main()
