# LR.PY PPRL implementation of the LR algorithm. 
# Author: Trong Nguyen 25/04/2026

# The goal is to implement Logistic Regression (LR) for Privacy-Preserving Record Linkage (PPRL) as a alternative to threshold based methods.

#PIPELINE:
#1. Read datasets and build block indices (same as existing code)   
#2. Encode records into Bloom filters and add DP noise (same as existing code)
#3. Match records using Logistic Regression instead of threshold-based matching (new code)

# PPRL - Privacy-preserving record linkage algorithm 
# using Bloom filter encoding with differential privacy
# phonetic blocking, and threshold-based classification. 
# Baseline methods - non-PPRL and PPRL without differential privacy
#
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

  # Logistic Regression implementation for PPRL
  # Added by: Trong Nguyen
  # Date: April 2026
  
  def match_lr(self, blk_index1, blk_index2, bf_dict1, bf_dict2, rec_dict1, rec_dict2):
    """Match and link records using Logistic Regression
    
    This method uses the already-encoded Bloom filters and applies
    Logistic Regression for classification instead of threshold-based matching.
    
    Args:
        blk_index1, blk_index2: Already-created block indices
        bf_dict1, bf_dict2: Already-encoded Bloom filters (with DP noise)
        rec_dict1, rec_dict2: Original records (for ground truth labels)
    
    Returns:
        Precision, Recall, F1 score
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score
    import numpy as np
    
    # Prepare features and labels
    pair_features = []
    pair_labels = []
    
    # Find common blocks (same as other methods)
    common_blks = []
    for blk in blk_index1:
        if blk in blk_index2:
            common_blks.append(blk)
    print('number of common blocks:', len(common_blks))
    
    # Compare all pairs within common blocks
    for blk in common_blks:
        recs_list1 = blk_index1[blk]
        recs_list2 = blk_index2[blk]
        for rec1 in recs_list1:
            for rec2 in recs_list2:
                # Calculate similarity using ALREADY-ENCODED Bloom filters
                bf1 = bf_dict1[rec1]
                bf2 = bf_dict2[rec2]
                sim = self.bf.calc_bf_sim(bf1, bf2)  # Uses existing BF similarity function
                pair_features.append([sim])
                
                # Create label from ground truth
                if rec_dict1[rec1][self.ent_id] == rec_dict2[rec2][self.ent_id]:
                    pair_labels.append(1)  # Match
                else:
                    pair_labels.append(0)  # Non-match
    
    # Convert to numpy arrays
    X = np.array(pair_features)
    y = np.array(pair_labels)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train Logistic Regression model
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = lr_model.predict(X_test)
    
    # Evaluate using sklearn metrics
    lr_precision = precision_score(y_test, y_pred, zero_division=0)
    lr_recall = recall_score(y_test, y_pred, zero_division=0)
    lr_f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Print results (same format as other methods)
    print('Number of LR matching pairs:', len(y_pred))
    print('LR linkage quality')
    print('Precision: ', lr_precision)
    print('Recall: ', lr_recall)
    print('F1 score: ', lr_f1)
    
    return lr_precision, lr_recall, lr_f1

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
    