# -*- coding: utf-8 -*-

import logging
import os
import sys
import unicodedata
from gensim import corpora, models, similarities

def word_count(s):
    wordcount = 0
    start = True
    for c in s:      
        cat = unicodedata.category(c)
        if cat == 'Lo':        # Letter, other
            wordcount += 1       # each letter counted as a word
            start = True                       
        elif cat[0] == 'P':    # Some kind of punctuation
            wordcount += 1       # each punctation counted as a word
            start = True                       
        elif cat[0] == 'Z':    # Some kind of separator
            start = True
        else:                  # Everything else
            if start:
                wordcount += 1     # Only count at the start
            start = False  
    return wordcount  

def process_dic(texts):
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            if word_count(token)>1:
                frequency[token] += 1
    return [[token for token in text if frequency[token] > 1] for text in texts]

def get_corpus(dictionary):
    corpus = [dictionary.doc2bow(text) for text in texts]
    #corpora.MmCorpus.serialize('/tmp/sum.mm', corpus)
    return corpus

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    #print('starting...')

    texts = []
    for line in sys.stdin:
        texts.append(line.split())

    from pprint import pprint
    #pprint(process_dic(texts))

    ### Building dictionary ###
    dictionary = corpora.Dictionary(process_dic(texts))
    dictionary.save('sum.dict')
    #print(dictionary)

    ### Building corpus ###
    corpus = get_corpus(dictionary)
    
    ### Creating transformation ###
    #tfidf = models.TfidfModel(corpus)

    #model = models.hdpmodel.HdpModel(corpus, id2word=dictionary)
    model = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=7)
    for index,topic in model.show_topics():
        print(topic)
