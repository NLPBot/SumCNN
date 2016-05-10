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
from feat_helper import *

def get_sents(file_name,model):
    sents, pos_list = [], []
    with open(file_name) as data_file:    
        data = json.load(data_file)
    for x in data['data']['documents']['document']:
        # get actual sentence
        sent = ''
        for word in x['order']['lemma']['text']:
            if str(word) in model: 
                sent += word + ' '
        if sent=='': continue
        sents.append(sent)
        pos_list.append( float(x['position']['paragraph']['forward']) )
    return sents, pos_list

def rank(to_be_ranked,stopwords,model,file_name,topic_dict):
    sentList = []
    for (sent_a,pos_list_a) in to_be_ranked:
        score = 0.
        semanSim = float(get_semantic_score(model,topic_dict,sent_a,file_name[:5]))
        for (sent_b,pos_list_b) in to_be_ranked:
            ngramSim = get_sim(sent_a,sent_b,stopwords)
            between_semanSim = model.n_similarity(sent_a.split(),sent_b.split())
            if "array" in str(type(between_semanSim)): between_semanSim = 0.
            score += ( ngramSim + semanSim + between_semanSim )
        if score==0.: score = 0.01
        sentList.append( (sent_a,score,pos_list_a) )
    return sorted(sentList, key=lambda tup: tup[1])

if __name__ == '__main__':

    # set up
    stopwords = nltk.corpus.stopwords.words('english')
    model = load_word2vec()
    topic_dict = get_topics()
    sub_dirs = ['feat_docs_para','feat_docs_sent','feat_model_para','feat_model_sent']
    data_dir = os.path.join('data',sub_dirs[1])

    # read all files    
    file_to_tuple = {}
    for file_name in os.listdir(data_dir):
        path_to_file = 'data/'+sub_dirs[1]+'/'+file_name

        # get sentences for this file
        (sents,pos_list) = get_sents(path_to_file,model)
        if file_name[:6] not in file_to_tuple.keys():
            file_to_tuple[file_name[:6]] = (sents,pos_list)
        else:
            file_to_tuple[file_name[:6]][0].extend(sents)
            file_to_tuple[file_name[:6]][1].extend(pos_list)

    for file_name in file_to_tuple:
        print(' .'.join(file_to_tuple[file_name][0]))
        pickle.dump( ' .'.join(file_to_tuple[file_name][0]), open('result/'+file_name+'.pkl','wb') )

    """

    # iterate through all topic
    for file_name in file_to_tuple:
        #print(file_name)

        sents = file_to_tuple[file_name][0]
        pos_list = file_to_tuple[file_name][1]
        to_be_ranked = zip(sents,pos_list)

        # rank 
        result = rank(to_be_ranked,stopwords,model,file_name,topic_dict)

        #print(result)
        #print('Scores: '+str(result) )
        pickle.dump( result, open('result/'+file_name+'.pkl','wb') )
    
    
    """
    
    
    
    
    
    
    
    
    
