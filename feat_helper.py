import json
from pprint import pprint
import os, pickle, sys
from SentSim import *
import nltk
from nltk.corpus import stopwords
import gensim
import numpy as np
import xml.etree.ElementTree as ET
from ngram import *
import string

sum_terms = [ 'and', 'also', 'in fact', 'actually', 'nor', 'too', 'on the other hand', 
            'much less', 'additionally', 'further', 'furthermore', 'either', 'moreoever',
            'besides', 'such as', 'for example', 'particularly', 'in particular', 'for instance',
            'especially', 'for one thing', 'as', 'like', 'let alone', 'considering', 'concerning',
            'the fact that', 'importantly', 'similarly', 'in the same way', 'by the same token',
            'equally', 'likewise', 'namely', 'specifically', 'that is', 'however', 'while', 'whereas',
            'conversely', 'even more', 'above all', 'indeed', 'more importantly', 'nevertheless',
            'even though', 'nonetheless', 'although', 'despite', 'in spite of', 'still', 'yet',
            'on the other hand', 'regardless', 'although', 'though', 'at least', 'rather', 'for',
            'since', 'owing to' ]

def load_word2vec():
    model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)
    return model
    #model.save('Word2Vec')
    #return gensim.models.Word2Vec.load('Word2Vec')

def get_sim(sent,summary,stopwords):
    return get_ngram_sim(sent.split(),summary.split())

def get_semantic_score(model,topic_dict,sent,file_name):
    gold_sum = []
    corpus = ' '.join(list(topic_dict[file_name])).split()
    for word in (corpus):
        if str(word) in model: 
            gold_sum.append( word )
    score = model.n_similarity(gold_sum, sent.split())
    if "array" in str(type(score)):
        return 0.0
    return score
    
def read_gold_sum(model):
    sum_dict = {}
    gold_sum = ''
    data_dir = os.path.join('data','feat_other_sent')

    print('reading gold summaries...')
    for file_name in os.listdir(data_dir):
        with open('data/feat_other_sent/'+file_name) as data_file:
            data = json.load(data_file)
        for x in data['data']['documents']['document']:
            gold_sum = ''
            for word in x['order']['lemma']['text']:
                # get summary
                gold_sum += word + ' '
            if file_name[:5] not in sum_dict.keys(): 
                sum_dict[file_name[:5]] = [ gold_sum ]
            else:
                sum_dict[file_name[:5]].append( gold_sum )
    print('Done reading gold summaries...')
    return sum_dict

def get_ratio(e,total):
    return float(e)/float(total)

def get_score_label(score):
    return int(score*100)

def get_num_of_punc(sent_list,punc):
    count = 0
    for word in sent_list:
        if word in punc:
            count += 1
    return count

def contains_term(term,sent):
    return term in sent

def remove_punc(s):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in s if ch not in exclude)

def get_topics():
    topic_dict = {}
    data_dir = os.path.join('data')
    tree = ET.parse('data/UpdateSumm09_test_topics.xml')
    root = tree.getroot()
    for topic in root:
        id = topic.attrib['id'][:5]
        title = ' '.join(str(topic[0].text).split())
        narrative = ' '.join(str(topic[1].text).split())
        title = remove_punc(title)
        narrative = remove_punc(narrative)
        topic_dict[id] = (title,narrative)
        #print(id+' '+title+' '+narrative)
    return topic_dict


