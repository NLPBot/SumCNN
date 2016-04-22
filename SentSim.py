#!/opt/python-3.4/bin/python3.4

# LING 573, Spring 2016
# Sentence Similarity

import sys
import math
import operator
import numpy


# Calculate the dot product of two vectors.
#
def dot_prod(vec1, vec2):

   prod = float(0)
   for a_feat in vec1:
      if a_feat in vec2:
         prod += float(vec1[a_feat]) * float(vec2[a_feat])

   return prod
# end dot_prod


# Calculate the magnitude of a vector.
#
def mag(vec):

   sigma = float(0)
   for a_feat in vec:
      sigma += vec[a_feat]**2

   return math.sqrt(sigma)
# end mag


# Find the cosine similarity between two vectors.
#
def cos_sim(vec1, vec2):

   return float(dot_prod(vec1, vec2)/(mag(vec1)*mag(vec2)))
# end cos_sim


# Create a binary word vector from a string.
#
def bin_vec(string):

   str_vec = {}

   for a_word in string.lower().split():
      if a_word not in str_vec:
         str_vec[a_word] = 1

   return str_vec
# end bin_vec


# Create a frequency word vector from a string.
#
def freq_vec(string):

   str_vec = {}

   for a_word in string.lower().split():
      if a_word not in str_vec:
         str_vec[a_word] = 1
      else:
         str_vec[a_word] += 1

   return str_vec
# end freq_vec


# Find the similarity between two sentences.
#
def binary_similarity(sent1, sent2):

   vec1 = bin_vec(sent1)
   vec2 = bin_vec(sent2)

   return cos_sim(vec1, vec2)
#end sentence_similarity


# Find the similarity between two sentences.
#
def frequency_similarity(sent1, sent2):

   vec1 = freq_vec(sent1)
   vec2 = freq_vec(sent2)

   return cos_sim(vec1, vec2)
#end sentence_similarity


def euclidean_dist(x,y):   
    return numpy.sqrt(numpy.sum((x-y)**2))

if __name__ == '__main__':

   numargs = len(sys.argv)
   if numargs < 3:
      print("Usage: SentSim.py SENTENCE1 SENTENCE2")
      raise SystemExit
   else:
      sentence1 = sys.argv[1]
      sentence2 = sys.argv[2]

   print("Cosine similarity (binary vectors) = " + str(binary_similarity(sentence1, sentence2)))
   print("Cosine similarity (frequency vectors) = " + str(frequency_similarity(sentence1, sentence2)))
