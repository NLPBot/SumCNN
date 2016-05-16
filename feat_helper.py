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
from collections import *
from gensim import corpora, models, similarities

sum_terms = [ 'and', 'also', 'in fact', 'actually', 'nor', 'too', 'on the other hand', 
            'much less', 'additionally', 'further', 'furthermore', 'either', 'moreoever',
            'besides', 'such as', 'for example', 'particularly', 'in particular', 'for instance',
            'especially', 'for one thing', 'as', 'like', 'let alone', 'considering', 'concerning',
            'the fact that', 'importantly', 'similarly', 'in the same way', 'by the same token',
            'equally', 'likewise', 'namely', 'specifically', 'that is', 'however', 'while', 'whereas',
            'conversely', 'even more', 'above all', 'indeed', 'more importantly', 'nevertheless',
            'even though', 'nonetheless', 'although', 'despite', 'in spite of', 'still', 'yet',
            'on the other hand', 'regardless', 'although', 'though', 'at least', 'rather', 'for',
            'since', 'owing to', 'according to' ]

def remove_sum_terms(sentence):
    sentence = sentence.split()
    global sum_terms
    for term in sum_terms:
        if term in sentence:
            sentence.remove(term)
    return ' '.join(sentence)

def lDistance(firstString, secondString):
    if len(firstString) > len(secondString):
        firstString, secondString = secondString, firstString
    distances = range(len(firstString) + 1)
    for index2, char2 in enumerate(secondString):
        newDistances = [index2 + 1]
        for index1, char1 in enumerate(firstString):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1], distances[index1+1], newDistances[-1])))
        distances = newDistances
    return distances[-1]

def process_topic(s):
    topics = []
    for new_s in s.split('+'):
        s_1 = ''.join( [i for i in new_s if not i.isdigit()] )
        topics.append( s_1.replace('*','').replace('.','').strip() )
    return topics

def getTopics(documents):
    stopwords = nltk.corpus.stopwords.words('english')
    texts = [[word for word in document.lower().split() if word not in stopwords] for document in documents]
    
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    texts = [[token for token in text if frequency[token] > 1] for text in texts]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    #tfidf = models.TfidfModel(corpus)
    model = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=1)
    topics = process_topic(model.show_topics()[0][1])
    return topics

def remove_punc(s):
    return ' '.join(ch for ch in s if ch not in exclude)

def get_sents(path_to_file):
    sents = []
    exclude = set(string.punctuation)
    with open(path_to_file) as data_file:    
        data = json.load(data_file)
    for x in data['data']['documents']['document']:
        # get actual sentence
        sent = ''
        for word in x['order']['lemma']['text']:
            sent += word + ' '
        if sent=='': continue
        sents.append(remove_punc(sent))
    return sents

def get_documents():
    doc_dict = {}
    data_dir = os.path.join('data','feat_docs_sent')
    for file_name in os.listdir(data_dir):
        path_to_file = 'data/feat_docs_sent/'+file_name
        # get sentences for this file
        sents = get_sents(path_to_file)
        if file_name[:5] not in doc_dict.keys():
            doc_dict[file_name[:5]] = sents
        else:
            doc_dict[file_name[:5]].extend( sents )
        #print(doc_dict) 
    return doc_dict

def load_word2vec():
    model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)
    return model
    #model.save('Word2Vec')
    #return gensim.models.Word2Vec.load('Word2Vec')

def get_sim(sent,summary,stopwords):
    return get_ngram_sim(sent.split(),summary.split())
    #return lDistance(sent,summary)

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

def get_doc_topics_dict():
    doc_dict = get_documents()
    topic_dict = {}
    for key in doc_dict.keys():
        topic_dict[key] = getTopics(doc_dict[key])
    return (topic_dict)

if __name__ == '__main__':
    doc_dict = get_documents()
    topic_dict = {}
    for key in doc_dict.keys():
        topic_dict[key] = getTopics(doc_dict[key])
    print(topic_dict)


















