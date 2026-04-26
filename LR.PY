# LR.PY PPRL implementation of the LR algorithm. 
# Author: Trong Nguyen 25/04/2026

# The goal is to implement Logistic Regression (LR) for Privacy-Preserving Record Linkage (PPRL) as a alternative to threshold based methods.

# Notes : Load datasets → block similar records → encode into Bloom filters with noise → 
# compute similarity → train Logistic Regression → predict matches → evaluate results.

import time
import math
import numpy as np
from BF import BF
from PPRL import Link
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

class LR:

    def __init__(self, link):

        # Link PPRL module with LR model
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

        # Find common blocks
        common_blks = [blk for blk in blk_ind1 if blk in blk_ind2]
        print(f'Number of common blocks: {len(common_blks)}')

        for blk in common_blks:
            recs_list1 = blk_ind1[blk]
            recs_list2 = blk_ind2[blk]

            for rec1 in recs_list1:
                for rec2 in recs_list2:

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
        # Evaluate model performance as per our requirements (precision, recall, F1-score)
        return (
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
            f1_score(y_test, y_pred)
        )
        


  
