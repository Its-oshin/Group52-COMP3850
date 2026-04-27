# LR.PY PPRL implementation of the LR algorithm. 
# Author: Trong Nguyen 25/04/2026

# The goal is to implement Logistic Regression (LR) for Privacy-Preserving Record Linkage (PPRL) as a alternative to threshold based methods.

# Our pipeline (to be confirmed!): Load datasets → block similar records → encode into Bloom filters with noise → 
# compute similarity → train Logistic Regression → predict matches → evaluate results.

import time
import numpy as np
from PPRL import Link
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


class LR:

    def __init__(self, link):
        self.link = link
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver='lbfgs'
        )
        self.is_trained = False

    def prepare_training_data(self, blk_ind1, blk_ind2, pbf_dict1, pbf_dict2, db1, db2, ent_index):
        features = []
        labels = []

        common_blks = [blk for blk in blk_ind1 if blk in blk_ind2]
        print(f'Number of common blocks: {len(common_blks)}')

        for blk in common_blks:
            for rec1 in blk_ind1[blk]:
                for rec2 in blk_ind2[blk]:

                    bf1 = pbf_dict1[rec1]
                    bf2 = pbf_dict2[rec2]

                    sim = self.link.bf.calc_bf_sim(bf1, bf2)
                    features.append([sim])

                    if db1[rec1][ent_index] == db2[rec2][ent_index]:
                        labels.append(1)
                    else:
                        labels.append(0)

        print(f'Total pairs: {len(features)}')
        print(f'Matches: {sum(labels)}')

        return np.array(features), np.array(labels)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X_test):
        if not self.is_trained:
            raise ValueError("Model not trained yet.")
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred):
        return (
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
            f1_score(y_test, y_pred)
        )


def run_lr_linkage(alice_path, bob_path, dataset_name):
    print('='*70)
    print(f'LOGISTIC REGRESSION - {dataset_name}')
    print('='*70)

    BF_length = 1000
    BF_num_hash = 10
    BF_q_gram = 2
    min_sim_val = 0.8
    link_attrs = [1, 2, 3, 4]
    block_attrs = [2, 4]
    ent_index = 0
    epsilon = 7

    link = Link(BF_length, BF_num_hash, BF_q_gram, min_sim_val,
                link_attrs, block_attrs, ent_index, epsilon)

    db1 = link.read_database(alice_path)
    db2 = link.read_database(bob_path)

    blk_ind1 = link.build_BI(db1)
    blk_ind2 = link.build_BI(db2)

    bf_dict1, _ = link.data_encode(db1)
    bf_dict2, _ = link.data_encode(db2)

    pbf_dict1 = link.add_DP_noise(bf_dict1)
    pbf_dict2 = link.add_DP_noise(bf_dict2)

    lr_model = LR(link)

    X, y = lr_model.prepare_training_data(
        blk_ind1, blk_ind2, pbf_dict1, pbf_dict2, db1, db2, ent_index
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    lr_model.train(X_train, y_train)
    y_pred = lr_model.predict(X_test)

    precision, recall, f1 = lr_model.evaluate(y_test, y_pred)

    print(f'Precision: {precision:.6f}')
    print(f'Recall:    {recall:.6f}')
    print(f'F1 Score:  {f1:.6f}')

    return {
        'dataset': dataset_name,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def run_all_datasets():
    datasets = [
        ('100_corr_25', 'Datasets/Alice_numrec_100_corr_25.csv',
         'Datasets/Bob_numrec_100_corr_25.csv'),
        ('100_corr_50', 'Datasets/Alice_numrec_100_corr_50.csv',
         'Datasets/Bob_numrec_100_corr_50.csv'),
        ('100_non_corr', 'Datasets/Alice_numrec_100_non_corr.csv',
         'Datasets/Bob_numrec_100_non_corr.csv'),
        ('500_corr_25', 'Datasets/Alice_numrec_500_corr_25.csv',
         'Datasets/Bob_numrec_500_corr_25.csv'),
        ('500_corr_50', 'Datasets/Alice_numrec_500_corr_50.csv',
         'Datasets/Bob_numrec_500_corr_50.csv'),
        ('500_non_corr', 'Datasets/Alice_numrec_500_non_corr.csv',
         'Datasets/Bob_numrec_500_non_corr.csv'),
    ]

    results = []

    for dataset_name, alice_path, bob_path in datasets:
        result = run_lr_linkage(alice_path, bob_path, dataset_name)
        results.append(result)
        print()

    print('\nSUMMARY')
    print('-'*60)
    for r in results:
        print(f'{r["dataset"]}: F1={r["f1"]:.6f}')


if __name__ == '__main__':
    run_all_datasets()

  
