# -*- coding: utf8 -*-
import sys; reload(sys); sys.setdefaultencoding('utf8')
import unicodedata
from collections import defaultdict
import pickle

def print_sents(docs):
    for sent in docs:
        print(sent)

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

def retrieve():
    try:
        return pickle.load( open( 'keyterms.pkl', "rb" ) )
    except:
        return defaultdict(int)
def store(l):
    pickle.dump( l, open('keyterms.pkl','wb') )

def string2vecs(ents):
    entities = retrieve()
    for entity in ents:
        entities[entity] += 1

    all_entities = []
    for k,v in entities.items():
        if word_count(unicode(k, "utf-8"))>1 and v>1:
            all_entities.append(k)

    unique_entities = list(set(all_entities))
    store(entities)
    
