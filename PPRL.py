# PPRL - Privacy-preserving record linkage algorithm 
# using Bloom filter encoding with differential privacy
# phonetic blocking, and threshold-based classification. 
# Baseline methods - non-PPRL and PPRL without differential privacy
#
# DV, Aug 2023
# -----------------------------------------------------------------------------

# Imports
#
import math
import random
import gzip
import string
import difflib
import copy
from BF import BF #import the BF module
from bitarray import bitarray
import hashlib
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np




# -----------------------------------------------------------------------------

#The class to read datasets, block, encode, perturb, match and link datasets, and evalaute the performance.
#No need to modify this section of the code
#
class Link:

  def __init__(self, length=1000, num_hash_func=30, q=2, min_sim_val=0.8, use_attr_index=12, blk_attr_index=2, ent_id=0, epsilon=5):
    """Constructor. Initialise an index, set common parameters.

       Arguments:
       - length                Number of bits used in the Bloom filter
       - num_hash_func         Number of hash functions used to map the q-grams in the Bloom filter
       - q                     Number of grams to split in the String
       - min_sim_val           Minimum similarity threshold for a comparison,
                               if below then a comparison result will be assigned
                               as the default bin '0'.
       - use_attr_index        A list of the index numbers of the attributes
                               that are used in the linking.
       - blk_attr_index        A list of the index numbers of the attributes
                               that are used in the blocking.
       - ent_id                Index of entity identifier - ground truth used for evaluation
       - epsilon               Privacy budget for differentially private Bloom filters
    """


    self.length = length
    self.num_hash_func = num_hash_func
    self.q = q
    self.min_sim_val = min_sim_val
    self.use_attr_index =  use_attr_index
    self.blk_attr_index =  blk_attr_index
    self.ent_id = ent_id
    self.epsilon = epsilon
    self.num_parties = 2 #currently suports only 2 party linkage
    
    #Create an instace for the BF class with the BF parameters
    self.bf = BF(self.length,self.num_hash_func,self.q)    
    
#####################################################

  def read_database(self, file_name):
    """Load the dataset and store in memory as a dictionary.
    """

    print('Load data file: %s' % (file_name))

    rec_dict = {}

    if (file_name.lower().endswith('.gz')):
      in_file =  gzip.open(file_name)  # Open gzipped file
    else:
      in_file =  open(file_name)  # Open normal file

    self.header_line = in_file.readline()  # Skip over header line

    rec_count = 0
    for rec in in_file:
      rec = rec.lower()
      clean_rec = rec.split(',')

      rec_id = str(rec_count)  # Assign unique number as record identifier
      assert rec_id not in rec_dict, ('Record ID not unique:', rec_id)

      # Save the original record with the record identifier
      #
      rec_dict[rec_id] = clean_rec
      rec_count += 1

    print('Read %d records' % (len(rec_dict)))
    return rec_dict

  # ---------------------------------------------------------------------------
    
  def get_soundex(self,value):
    """Get the soundex code for the string"""
    value = value.upper()

    soundex = ""
    soundex += value[0]

    dictionary = {"BFPV": "1", "CGJKQSXZ":"2", "DT":"3", "L":"4", "MN":"5", "R":"6", "AEIOUHWY":"."}

    for char in value[1:]:
        for key in dictionary.keys():
            if char in key:
                code = dictionary[key]
                if code != soundex[-1]:
                    soundex += code

    soundex = soundex.replace(".", "")
    soundex = soundex[:4].ljust(4, "0")

    return soundex

  # ---------------------------------------------------------------------------

  def build_BI(self, rec_dict):
    """Build Block Index to store the blocking key values and the corresponding list of record identifiers.
    """
    block_index = {}
    blk_attr_index = self.blk_attr_index

    print('Build Block Index for attributes:', blk_attr_index)

    for (rec_id, clean_rec) in rec_dict.items():
      compound_bkv = ""
      for attr in blk_attr_index:  # Process selected blocking attributes
        attr_val = clean_rec[attr]
        attr_encode = self.get_soundex(attr_val)
        compound_bkv += attr_encode
      
      if (compound_bkv in block_index): 	 # Block value in index, only add attribute value
        rec_id_list = block_index[compound_bkv]
        rec_id_list.append(rec_id)
      else: # A new block, add block value and attribute value
        rec_id_list = [rec_id]
        block_index[compound_bkv] = rec_id_list

    print('Generate %d blocks' % (len(block_index)))
    return block_index

  # ---------------------------------------------------------------------------

  def string_similarity(self, str1, str2):
    """Calculate similarity between string values (baseline non-privacy-preserving linkage)
    """
    result =  difflib.SequenceMatcher(a=str1.lower(), b=str2.lower())
    return result.ratio()

  # ---------------------------------------------------------------------------

  def data_encode(self,rec_dict):
    """Encode records into Bloom filters.
    """
    BF_dict = {}
    
    all_val_set = [] #contains all vals (e.g. q-grams) - required to calculate the false positive rate
        
    for rec in rec_dict:
        this_rec_list = rec_dict[rec]
        this_rec_bf = bitarray(self.length)
        this_rec_bf.setall(False)
        for attr in self.use_attr_index:
            this_attr_val_set = self.bf.convert_str_val_to_set(this_rec_list[attr])
            all_val_set += this_attr_val_set
            this_attr_bf = self.bf.set_to_bloom_filter(this_attr_val_set)
            this_rec_bf |= this_attr_bf
        BF_dict[rec] = this_rec_bf
    
    return BF_dict, all_val_set

  # ---------------------------------------------------------------------------

  def add_DP_noise(self,bf_dict):
    """Encode records into Bloom filters.
    """
            
    #print(bf_dict)
    pbf_dict = copy.deepcopy(bf_dict) #copy the Bloom filter dictionary
    
    #perturb the copied Bloom filter dictionary to add noise
    for bf in pbf_dict:
        this_bf = pbf_dict[bf]
        prob_to_flip = 1.0/(1+math.e**(self.epsilon/2))
        num_bits_to_flip = int(self.length * prob_to_flip)
        indices = [x for x in range(self.length)]
        indices_to_flip = random.sample(indices, num_bits_to_flip)
        
        for ind in indices_to_flip:
            if this_bf[ind] == 0:
                this_bf[ind] = 1
            else:
                this_bf[ind] = 0
                
    #print(bf_dict)
        
    return pbf_dict    

  # ---------------------------------------------------------------------------

  def match(self, blk_index1, blk_index2, bf_dict1, bf_dict2):
    """Match and link records based on the similarity between their Bloom filter encodings.
    """
    matches = []
    common_blks = []
    
    for blk in blk_index1:
        if blk in blk_index2:
            common_blks.append(blk)
    print('number of common blocks:', len(common_blks))
    
    for blk in common_blks:
        recs_list1 = blk_index1[blk]
        recs_list2 = blk_index2[blk]
        for rec1 in recs_list1:
            for rec2 in recs_list2:
                bf1 = bf_dict1[rec1]
                bf2 = bf_dict2[rec2]
                sim = self.bf.calc_bf_sim(bf1,bf2)
                if sim >= self.min_sim_val:
                    matches.append([rec1,rec2])
    print('Number of matching pairs:', len(matches))
    #print(matches)
    return matches

  # ---------------------------------------------------------------------------

  def match_npp(self, blk_index1, blk_index2, rec_dict1, rec_dict2):
    """Match and link records based on the similarity between their attributes (non-privacy-preserving).
    """
    matches = []
    common_blks = []
    
    for blk in blk_index1:
        if blk in blk_index2:
            common_blks.append(blk)
    print('number of common blocks:', len(common_blks))
    
    for blk in common_blks:
        recs_list1 = blk_index1[blk]
        recs_list2 = blk_index2[blk]
        for rec1 in recs_list1:
            for rec2 in recs_list2:
                #Baseline non-privacy-preserving linkage method
                tot_sim = 0.0
                for attr in self.use_attr_index:
                    tot_sim += self.string_similarity(rec_dict1[rec1][attr],rec_dict2[rec2][attr])
                if tot_sim/len(self.use_attr_index) >= self.min_sim_val:
                    matches.append([rec1,rec2])
    print('Number of matching pairs:', len(matches))
    #print(matches)
    return matches

  # ---------------------------------------------------------------------------

  def evaluate(self, matches, rec_dict1, rec_dict2):
    """Evaluate linkage quality.
    """    
    num_matches = len(matches)
    num_true_matches = 0
    num_all_true_matches = 0
        
    for rec1 in rec_dict1:
        for rec2 in rec_dict2:
            if rec_dict1[rec1][self.ent_id] == rec_dict2[rec2][self.ent_id]:
                num_all_true_matches += 1
    
    for match in matches:
        if rec_dict1[match[0]][self.ent_id] == rec_dict2[match[1]][self.ent_id]:
            num_true_matches += 1
            
    #print(num_matches, num_true_matches, num_all_true_matches)
    
    if num_matches > 0.0:
        precision = num_true_matches/num_matches
    else:
        precision = 0.0
    if num_all_true_matches > 0.0:
        recall = num_true_matches/num_all_true_matches
    else:
        recall = 0.0
    if (precision+recall) > 0.0:
        f1score = (2*precision*recall)/(precision+recall)
    else:
        f1score = 0.0
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1 score: ', f1score)

    return precision, recall, f1score

  # ---------------------------------------------------------------------------

  #Generate linkage quality comparison graph
  def gen_acc_plot(self, orig_list, base1_list, base2_list):

    X1 = [0,4,8]
    X2 = [1,5,9]
    X3 = [2,6,10]

    plt.bar([0,4,8], [0.8, 0.9, 0.95], color='red', label = 'Original') 
    plt.bar(X2, base1_list, color='#9467bd', label = 'Baseline 1') 
    plt.bar(X3, base2_list, color='#2300A8', label = 'Baseline 2') 

    plt.text(-0.4, orig_list[0], str(round(orig_list[0],3)))
    plt.text(0.6, base1_list[0], str(round(base1_list[0],3)))
    plt.text(1.6, base2_list[0], str(round(base2_list[0],3)))
    plt.text(3.6, orig_list[1], str(round(orig_list[1],3)))
    plt.text(4.6, base1_list[1], str(round(base1_list[1],3)))
    plt.text(5.6, base2_list[1], str(round(base2_list[1],3)))
    plt.text(7.6, orig_list[2], str(round(orig_list[2],3)))
    plt.text(8.6, base1_list[2], str(round(base1_list[2],3)))
    plt.text(9.6, base2_list[2], str(round(base2_list[2],3)))

    plt.title('Linkage quality comparison')
    plt.xticks([0.9,4.9,8.9], ['Precision', 'Recall', 'F1-score'], fontsize=10)
    plt.xlim(xmin=-0.9, xmax=10.9)
    plt.ylim(ymax=1.01,ymin=0.8)
    plt.legend(loc='best')
    plt.show()

  # ---------------------------------------------------------------------------

  #Generate privacy comparison graph
  def gen_priv_plot(self, orig_list, base1_list, base2_list):

    X1 = [0,4]
    X2 = [1,5]
    X3 = [2,6]

    plt.bar(X1, orig_list, color='#007ACC', label = 'Original') 
    plt.bar(X2, base1_list, color='#9467bd', label = 'Baseline 1') 
    plt.bar(X3, base2_list, color='#2300A8', label = 'Baseline 2') 

    plt.text(-0.4, orig_list[0], str(round(orig_list[0],3)))
    plt.text(0.6, base1_list[0], str(round(base1_list[0],3)))
    plt.text(1.6, base2_list[0], str(round(base2_list[0],3)))
    plt.text(3.6, orig_list[1], str(round(orig_list[1],3)))
    plt.text(4.6, base1_list[1], str(round(base1_list[1],3)))
    plt.text(5.6, base2_list[1], str(round(base2_list[1],3)))
  

    plt.title('Privacy comparison')
    plt.xticks([0.9,4.9], ['false positive rate', 'privacy budget'], fontsize=10)
    plt.xlim(xmin=-0.9, xmax=5.9)
    plt.ylim(ymin=0.0)
    plt.legend(loc='best')
    plt.show()
  
  # ---------------------------------------------------------------------------

  # Logistic Regression - Trong
   
  def match_lr(self, blk_index1, blk_index2, bf_dict1, bf_dict2, rec_dict1, rec_dict2):
    """Match and link records using Logistic Regression over Bloom filter similarities."""


    X = []
    y = []

    common_blks = [blk for blk in blk_index1 if blk in blk_index2]
    print('Number of common blocks:', len(common_blks))

    for blk in common_blks:
      recs1 = blk_index1[blk]
      recs2 = blk_index2[blk]
      for r1 in recs1:
        for r2 in recs2:
          bf1 = bf_dict1[r1]
          bf2 = bf_dict2[r2]
          sim = self.bf.calc_bf_sim(bf1, bf2)
          X.append([sim])
          if rec_dict1[r1][self.ent_id] == rec_dict2[r2][self.ent_id]:
            y.append(1)
          else:
            y.append(0)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)

    matches = []
    index = 0

    for blk in common_blks:
      recs1 = blk_index1[blk]
      recs2 = blk_index2[blk]
      for r1 in recs1:
        for r2 in recs2:
          prediction = model.predict([X[index]])[0]
          if prediction == 1:
            matches.append([r1, r2])
          index += 1

    print('Number of matching pairs:', len(matches))
    return matches

  # ---------------------------------------------------------------------------
  
  # Random Forest Linkage - Liza
  def match_rf(self, blk_index1, blk_index2, bf_dict1, bf_dict2, rec_dict1, rec_dict2):
    """Match and link records using Random Forest classifier."""


    pair_features = []
    pair_labels = []

    for blk in blk_index1:
        if blk in blk_index2:
            for rec1 in blk_index1[blk]:
                for rec2 in blk_index2[blk]:
                    bf1 = bf_dict1[rec1]
                    bf2 = bf_dict2[rec2]
                    sim = self.bf.calc_bf_sim(bf1, bf2)
                    pair_features.append([sim])
                    if rec_dict1[rec1][0] == rec_dict2[rec2][0]:
                        pair_labels.append(1)
                    else:
                        pair_labels.append(0)

    X = np.array(pair_features)
    y = np.array(pair_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    rf_precision = precision_score(y_test, y_pred, zero_division=0)
    rf_recall = recall_score(y_test, y_pred, zero_division=0)
    rf_f1 = f1_score(y_test, y_pred, zero_division=0)

    matches = []
    for i, pred in enumerate(y_pred):
        if pred == 1:
            matches.append(i)

    return matches, rf_precision, rf_recall, rf_f1
  
 # ---------------------------------------------------------------------------

  #SVM Linkage - Jeslin 
  def match_svm(self, blk_index1, blk_index2, bf_dict1, bf_dict2, rec_dict1, rec_dict2):

    X = []
    y = []

    common_blks = [blk for blk in blk_index1 if blk in blk_index2]

    for blk in common_blks:
        recs1 = blk_index1[blk]
        recs2 = blk_index2[blk]

        for r1 in recs1:
            for r2 in recs2:

                bf1 = bf_dict1[r1]
                bf2 = bf_dict2[r2]

                sim = self.bf.calc_bf_sim(bf1, bf2)

                X.append([sim])

                if rec_dict1[r1][self.ent_id] == rec_dict2[r2][self.ent_id]:
                    y.append(1)
                else:
                    y.append(0)

    model = svm.SVC(kernel='linear')
    model.fit(X, y)

    matches = []
    index = 0

    for blk in common_blks:
        recs1 = blk_index1[blk]
        recs2 = blk_index2[blk]

        for r1 in recs1:
            for r2 in recs2:

                prediction = model.predict([X[index]])[0]

                if prediction == 1:
                    matches.append([r1, r2])

                index += 1

    print('Number of matching pairs (SVM):', len(matches))

    return matches
    
  # ---------------------------------------------------------------------------
     
  # KNN based record linkage using Bloom filter similarities - Xai
  # build candidate pairs from shared blocks
  def _build_candidate_pairs(self, blk_index1, blk_index2):
    """Build candidate record pairs from shared blocks."""
    common_blks = []
    candidate_pairs = []

    for blk in blk_index1:
      if blk in blk_index2:
        common_blks.append(blk)

    for blk in common_blks:
      recs_list1 = blk_index1[blk]
      recs_list2 = blk_index2[blk]
      for rec1 in recs_list1:
        for rec2 in recs_list2:
          candidate_pairs.append([rec1, rec2])

    return common_blks, candidate_pairs

  # ---------------------------------------------------------------------------
  # balance positive and negative examples for KNN training
  def _balance_training_examples(self, pos_examples, neg_examples, balance):
    """Optionally rebalance positive and negative training examples."""
    if balance != True or len(pos_examples) == 0 or len(neg_examples) == 0:
      return pos_examples, neg_examples

    if len(neg_examples) > len(pos_examples):
      neg_examples = random.sample(neg_examples, len(pos_examples))
    elif len(pos_examples) > len(neg_examples):
      pos_examples = random.sample(pos_examples, len(neg_examples))

    return pos_examples, neg_examples

  # ---------------------------------------------------------------------------
  # build KNN training set from candidate pairs and bloom filter similarities
  def _build_knn_training(self, candidate_pairs, rec_dict1, rec_dict2, bf_dict1,
                          bf_dict2, sample_limit=None, balance=True):
    """Build a 1D KNN training set using Bloom filter similarities."""
    pos = []
    neg = []

    for rec1, rec2 in candidate_pairs:
      bf1 = bf_dict1[rec1]
      bf2 = bf_dict2[rec2]
      sim = self.bf.calc_bf_sim(bf1, bf2)
      label = 1 if rec_dict1[rec1][self.ent_id] == rec_dict2[rec2][self.ent_id] else 0

      if label == 1:
        pos.append((sim, label))
      else:
        neg.append((sim, label))

    if len(pos) == 0 and len(neg) == 0:
      raise ValueError('No candidate pairs found for training')

    pos, neg = self._balance_training_examples(pos, neg, balance)

    sampled = pos + neg
    random.shuffle(sampled)

    if sample_limit is not None and sample_limit < len(sampled):
      sampled = sampled[:sample_limit]

    if len(sampled) == 0:
      raise ValueError('Training set is empty after sampling')

    train_sims = [sim for sim, _label in sampled]
    train_labels = [label for _sim, label in sampled]
    return train_sims, train_labels

  # ---------------------------------------------------------------------------

  # KNN based matching using bloom filter similarities and trained KNN model
  def match_knn(self, blk_index1, blk_index2, bf_dict1, bf_dict2, rec_dict1,
                rec_dict2, k=3, sample_limit=None, balance=True):
    """Match records using KNN over Bloom filter similarity scores."""
    common_blks, candidate_pairs = self._build_candidate_pairs(blk_index1, blk_index2)
    print('Number of common blocks:', len(common_blks))

    train_sims, train_labels = self._build_knn_training(candidate_pairs, rec_dict1,
                                                        rec_dict2, bf_dict1,
                                                        bf_dict2, sample_limit,
                                                        balance)

    k = max(1, min(int(k), len(train_sims)))
    matches = []

    for rec1, rec2 in candidate_pairs:
      bf1 = bf_dict1[rec1]
      bf2 = bf_dict2[rec2]
      sim = self.bf.calc_bf_sim(bf1, bf2)

      dists = [(abs(sim - train_sim), idx) for idx, train_sim in enumerate(train_sims)]
      dists.sort(key=lambda item: item[0])
      votes = [train_labels[idx] for (_dist, idx) in dists[:k]]
      pred = 1 if sum(votes) >= (len(votes) / 2.0) else 0

      if pred == 1:
        matches.append([rec1, rec2])

    print('Number of matching pairs:', len(matches))
    return matches
  
##########################################################
