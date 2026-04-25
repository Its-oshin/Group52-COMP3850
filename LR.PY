# LR.PY PPRL implementation of the LR algorithm. 
# Author: Trong Nguyen 25/04/2026

# The goal is to implement Logistic Regression (LR) for Privacy-Preserving Record Linkage (PPRL) as a alternative to threshold based methods.

# Notes : Load datasets → block similar records → encode into Bloom filters with noise → 
# compute similarity → train Logistic Regression → predict matches → evaluate results.

import time
import math
from BF import BF
from PPRL import Link
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

class LR:

def __init__(self, link):

# we are linking the PPRL moduel with the LR class
# And link is the object of link class which has the data and metrics 

  self.link = link
  self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver='lbfgs'
        )

self.is_trained = False

def prepare_training_data(self, blk_ind1, blk_ind2, pbf_dict1, pbf_dict2, 
                              db1, db2, ent_index):
  # Just getting the data ready for training
  


  
