# BF - Bloom filter program to convert string, numerical 
# (integer, floating point) and modulus values that have finitie range
# (e.g., longitude and latitude with range 0-360 degrees) into Bloom filters to 
# allow privacy-preserving similarity calculations.
#
# DV, Mar 2015
# -----------------------------------------------------------------------------

# imports
#

# Standard Python modules
import math
import random
import hashlib
import gzip
import os


random.seed(9318)

from bitarray import bitarray
import matplotlib
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------

class BF():

  def __init__(self, bf_len, bf_num_hash_func, q, bf_num_inter=2, bf_step=1, 
               max_abs_diff=5, min_val=0, max_val=100):
    """Initialisation, set class parameters:
       - bf_len            Length of Bloom filters
       - bf_num_hash_func  Number of hash functions
       - bf_num_interval   Number of intervals to use for BF based similarities

       - max_abs_diff      Maximum absolute difference allowed
       - min_val           Minimum value 
       - max_val           Maximum value
       - q                 Length of sub-strings (q-grams)
    """

    self.bf_len =           bf_len
    self.bf_num_hash_func = bf_num_hash_func
    self.bf_num_inter =     bf_num_inter
    self.bf_step = 	    bf_step

    self.max_abs_diff =  max_abs_diff
    self.min_val =       min_val
    self.max_val =       max_val

    self.q = q

    assert max_val > min_val

    # Bloom filter shortcuts
    #
    self.h1 = hashlib.sha1
    self.h2 = hashlib.md5

  # ---------------------------------------------------------------------------

  def set_to_bloom_filter(self, val_set):
    """Convert an input set of values into a Bloom filter.
    """

    k = self.bf_num_hash_func
    l = self.bf_len

    bloom_set = bitarray(l)  # Convert set into a bit array		
    bloom_set.setall(False)
         
    for val in val_set:
      hex_str1 = self.h1(val.encode('utf-8')).hexdigest() #self.h1(val).hexdigest()
      int1 =     int(hex_str1, 16)
      hex_str2 = self.h2(val.encode('utf-8')).hexdigest() #self.h2(val).hexdigest()
      int2 =     int(hex_str2, 16)

      for i in range(k):
        gi = int1 + i*int2
        gi = int(gi % l)
        bloom_set[gi] = True
      
    return bloom_set

  # ---------------------------------------------------------------------------

  def calc_bf_sim(self, bf1, bf2):
    """Calculate Dice coefficient similarity of two Bloom filters.
    """

    bf1_1s = bf1.count()
    bf2_1s = bf2.count()

    common_1s = (bf1 & bf2).count()

    dice_sim = (2.0 * common_1s)/(bf1_1s + bf2_1s)
      
    return dice_sim

  # ---------------------------------------------------------------------------

  def calc_abs_diff(self, val1, val2):
    """Calculate absolute difference similarity between two values based on the
       approach described in:

       Data Matching, P Christen, Springer 2012, page 121, equations (5.28).
    """

    max_abs_diff = self.max_abs_diff

    if (val1 == val2):
      return 1.0

    abs_val_diff = abs(float(val1) - float(val2))

    if (abs_val_diff >= max_abs_diff):
      return 0.0  # Outside allowed maximum difference

    abs_sim = 1.0 - abs_val_diff / max_abs_diff

    assert abs_sim > 0.0 and abs_sim < 1.0, (val1, val2, abs_sim)

    return abs_sim

  # ---------------------------------------------------------------------------

  def calc_str_sim(self, val1, val2):
    """Calculate dice-coefficient similarity between two strings (non-encoded).
    """
    # Length of sub-strings (q-grams) to be extracted from string values
    #
    q = self.q # q-gram length

    val1_set = [val1[i:i+q] for i in range(len(val1) - (q-1))]  
    val2_set = [val2[i:i+q] for i in range(len(val2) - (q-1))]

    num_items_val1 = len(list(set(val1_set)))
    num_items_val2 = len(list(set(val2_set)))
    num_common_items = len(list(set(val1_set) & set(val2_set)))

    dice_sim = (2.0 * num_common_items) / (num_items_val1 + num_items_val2)

    return dice_sim

  # ---------------------------------------------------------------------------

  def calc_cate_sim(self, val1, val2):
    """Calculate similarity between two categorical values (non-encoded).
       - exact matching
    """
    if val1 == val2:
      sim = 1.0
    else:
      sim = 0.0

    return sim

  # ---------------------------------------------------------------------------

  def convert_str_val_to_set(self, val):
    """Covert string values into lists to be hash-mapped into the Bloom filters.
    """

    # Length of sub-strings (q-grams) to be extracted from string values
    #
    q = self.q # q-gram length

    val_set = [val[i:i+q] for i in range(len(val) - (q-1))]  

    return val_set


  # ---------------------------------------------------------------------------

  def convert_num_val_to_set(self, val):
    """Covert numeric values into lists to be hash-mapped into the Bloom filters.
    """

    # Number of intervals and their sizes (step) to consider
    #
    bf_num_inter = self.bf_num_inter
    bf_step =           self.bf_step

    val_set = set()

    rem_val = val % bf_step  # Convert into values within same interval
    if rem_val >= bf_step/2:
      use_val = val + (bf_step - rem_val)
    else:
      use_val = val - rem_val


    val_set.add(str(float(use_val)))  # Add the actual value

    # Add variations larger and smaller than the actual value
    #
    for i in range(bf_num_inter+1):
      diff_val = (i+1)*bf_step
      val_set.add(str(use_val - diff_val))

      diff_val = (i)*bf_step
      val_set.add(str(use_val + diff_val))

    return val_set

  # ---------------------------------------------------------------------------

  def calc_mod_diff(self, val1, val2):
    """Calculate difference similarity between two modulus values that have finite range
       (in contrast to integer and floating point values that have infinite range).
    """

    max_abs_diff = self.max_abs_diff
    min_val = self.min_val
    max_val = self.max_val

    if (val1 == val2):
      return 1.0

    mod_val_diff = float((max_val - max(val1,val2)) + (min(val1,val2)-min_val)+1)
    #print mod_val_diff
    if (mod_val_diff >= max_abs_diff):
      return 0.0  # Outside allowed maximum difference

    mod_sim = 1.0 - mod_val_diff / max_abs_diff

    assert mod_sim > 0.0 and mod_sim < 1.0, (val1, val2, mod_sim)

    return mod_sim

  # ---------------------------------------------------------------------------

  def convert_mod_val_to_set(self, val1):
    """Convert modulus values into sets to be hash-mapped
       into Bloom filters.
    """

    # Number of intervals and their sizes (step) to consider
    #
    bf_num_inter = self.bf_num_inter
    bf_step =           self.bf_step

    min_val = self.min_val
    max_val = self.max_val

    val_set = set()

    rem_val = val % bf_step  # Convert into values within same interval
    if rem_val >= bf_step/2:
      use_val = val + (bf_step - rem_val)
    else:
      use_val = val - rem_val

    val_set.add(str(float(use_val)))  # Add the actual values

    # Add variations larger and smaller than the actual value
    #
    for i in range(bf_num_inter+1):

      diff_val = (i+1)*bf_step
      prev_val = use_val - diff_val
      if prev_val < min_val:
        val_set.add(str(prev_val + (max_val-min_val+1)))
      else:
        val_set.add(str(prev_val))

      diff_val = (i)*bf_step
      next_val = use_val + diff_val
      if next_val > max_val:
        val_set.add(str(next_val%(max_val-min_val)))
      else:
        val_set.add(str(next_val))


    return val_set

  # ---------------------------------------------------------------------------

##########################################################
