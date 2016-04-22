#!/opt/python-3.4/bin/python3.4

# LING 573, Spring 2016
# Information Ordering

import sys
import operator


LOWER_IS_BETTER = False
HIGHER_IS_BETTER = True
SORT_ORDER = LOWER_IS_BETTER

def order_summary(input):

   scores = []
   summary = {}
   ordered_summary = []

   for i, inline in enumerate(input):
      linelist = inline.split()
      score = float(linelist.pop(0))
      sentence = " ".join(linelist)
      scores.append([i, score])
      summary[i] = sentence
   
   scores.sort(key=operator.itemgetter(1), reverse=SORT_ORDER)
   for a_score in scores:
      ordered_summary.append(summary[a_score[0]])

   return ordered_summary
#end order_summary



if __name__ == '__main__':

   numargs = len(sys.argv)
   if numargs == 2:
      print("Usage: InfoOrder.py INFILE OUTFILE")
      raise SystemExit
   elif numargs > 2:
      infile = open(sys.argv[1])
      outfile = open(sys.argv[2], "w")
   else:
      infile = sys.stdin
      outfile = sys.stdout

   inlist = []

   for a_line in infile:
      inlist.append(a_line)
   outlist = order_summary(inlist)
   if numargs > 2: infile.close()

   for a_sentence in outlist:
      outfile.write(a_sentence + "\n")
   if numargs > 2: outfile.close()
