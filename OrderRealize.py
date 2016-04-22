#!/opt/python-3.4/bin/python3.4

# LING 573, Spring 2016
# Information Ordering and Content Realization

import os
import sys
import math
import pickle
import operator
import optparse


WC_LIMIT = 100
#INTERIM_MAX = int(WC_LIMIT/1)
#REDUNDANCY_MAX = 0.5
LOWER_IS_BETTER = False
HIGHER_IS_BETTER = True
SORT_ORDER = LOWER_IS_BETTER
PUNCT = '.,;:\'\"?!'



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



def order_summary(topic):

   interim_summary = []
   ordered_summary = []

# Sort the topic sentences by score where higher is better.
#
   topic.sort(key=operator.itemgetter(1), reverse=HIGHER_IS_BETTER)

# Check for redundancy before adding a sentence to the summary.
#
   for sent_idx, sent_data in enumerate(topic):
      if sent_idx == 0:
         interim_summary.append((sent_data[0], sent_data[2]))
      else:
         addit = True
         for summ_sent in interim_summary:
            if frequency_similarity(sent_data[0], summ_sent[0]) > REDUNDANCY_MAX:
               addit = False
               break
         if addit and (len(interim_summary) < INTERIM_MAX):
            interim_summary.append((sent_data[0], sent_data[2]))
         elif len(interim_summary) >= INTERIM_MAX:
            break

   interim_summary.sort(key=operator.itemgetter(1), reverse=LOWER_IS_BETTER)

   for summ_sent in interim_summary:
      ordered_summary.append(summ_sent[0])

   return ordered_summary
#end order_summary


def detokenize(sent_str):

   str_list = list(sent_str)

   prev_char = ""
   for idx, char in enumerate(str_list):
      if (char in PUNCT) and (prev_char == ' '):
         str_list.pop(idx - 1)
      prev_char = char

   return "".join(str_list)
# end detokenize
      

def realize_summary(input):

   outlist = []

   summary_word_count = 0
   for inline in input:
      lessline = detokenize(inline)
      linelist = lessline.split()
      if (len(linelist) + summary_word_count) <= WC_LIMIT:
         outlist.append(lessline)
         summary_word_count += len(linelist)
   return outlist


if __name__ == '__main__':


   cmd = optparse.OptionParser()
   cmd.add_option("-c", action="store", dest="compress")
   cmd.add_option("--compression", action="store", dest="compress")
   cmd.set_defaults(compress=100)
   cmd.add_option("-r", action="store", dest="redundancy")
   cmd.add_option("--redundancy", action="store", dest="redundancy")
   cmd.set_defaults(redundancy=0.5)
   opts, args = cmd.parse_args()
   
   numargs = len(args)
   if numargs != 2:
      sys.stderr.write("Usage: %s INDIR OUTDIR [-c 100 -r 0.5]\n" % sys.argv[0])
      raise SystemExit(1)
   else:
      indir = args[0]
      outdir = args[1]

   INTERIM_MAX = int(opts.compress)
   REDUNDANCY_MAX = float(opts.redundancy)

   topic_sents = {}
   topic_summs = {}

   print("Compression = %d sentences." % INTERIM_MAX)
   print("Redundancy threshold = %f." % REDUNDANCY_MAX)

# Load the topic sentences along with scores and
# position values from the pickle files created
# by the content selection/scoring module. There
# is a file per document not per topic.
#
   for resfile in os.listdir(indir):
      if resfile[0:2] != '._':
         topic_id = resfile[0:6]
         if topic_id not in topic_sents:
            topic_sents[topic_id] = pickle.load(open(indir+"/"+resfile, "rb"))
         else:
            topic_sents[topic_id].extend(pickle.load(open(indir+"/"+resfile, "rb")))

   for a_topic in topic_sents:
      sys.stderr.write("Topic: " + a_topic + "\r")
      topic_summs[a_topic] = realize_summary(order_summary(topic_sents[a_topic]))

   for a_topic in topic_summs:
      id_part1 = a_topic[0:5]
      id_part2 = a_topic[5:6]
      summ_path = outdir + '/' + id_part1 + '-A.M.100.' + id_part2 + '.' + 'UI' + id_part1
      summ_fd = open(summ_path, "w")
      for a_sent in topic_summs[a_topic]:
         summ_fd.write(a_sent + "\n")
      summ_fd.close()

