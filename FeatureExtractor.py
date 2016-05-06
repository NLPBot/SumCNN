import json
from pprint import pprint
import os, pickle, sys
from SentSim import *
import nltk
from nltk.corpus import stopwords
import gensim
import numpy as np
from numpy import *
import xml.etree.ElementTree as ET
from ngram import *
import string
from feat_helper import *

def get_data_pair(predict=False):
    global feat_num, sum_terms

    # setting up
    feat_vec_list, scores, word_vec_list = [], [], []
    score = 0
    
    model = load_word2vec()
    topic_dict = get_topics()
    stopwords = nltk.corpus.stopwords.words('english')
    if not(predict):     
        sum_dict = read_gold_sum(model)
        
    sub_dirs = ['feat_docs_para','feat_docs_sent','feat_model_para','feat_model_sent']
    data_dir = os.path.join('data',sub_dirs[1])
    for file_name in os.listdir(data_dir):
        with open('data/'+sub_dirs[1]+'/'+file_name) as data_file:    
            data = json.load(data_file)

        print('within file '+file_name[:5])
        pos_list, p_feat_vec_list, pre_actual_sent_list = [], [], []

        for x in data['data']['documents']['document']:
            # append feature list
            feat_vec = []
            
            # 20 features
            feat_vec.append(x['conj']['+'])
            feat_vec.append(x['conj']['-'])
            feat_vec.append(x['count']['punct'])
            feat_vec.append(x['count']['stop0'])
            feat_vec.append(x['count']['stop1'])
            feat_vec.append(x['length'])
            feat_vec.append(x['overlap'])
            pos_list.append( float(x['position']['paragraph']['forward']) )
            total = float(x['position']['paragraph']['forward']) + float(x['position']['paragraph']['reverse'])            
            feat_vec.append(get_ratio(float(x['position']['paragraph']['forward']),total))
            total = float(x['position']['sentence']['document']['forward']) + float(x['position']['sentence']['document']['reverse'])
            feat_vec.append(get_ratio(float(x['position']['sentence']['document']['forward']),total))
            total = float(x['position']['sentence']['paragraph']['forward']) + float(x['position']['sentence']['paragraph']['reverse'])
            feat_vec.append(get_ratio(float(x['position']['sentence']['paragraph']['forward']),total))
            feat_vec.append(x['ratio']['punct']['word'][0])
            feat_vec.append(x['ratio']['punct']['word'][1])
            feat_vec.append(get_num_of_punc(x['order']['word']['text'],'"'))
            feat_vec.append(get_num_of_punc(x['order']['word']['text'],'.'))
            feat_vec.append(get_num_of_punc(x['order']['word']['text'],'?'))
            feat_vec.append(get_num_of_punc(x['order']['word']['text'],':'))
            feat_vec.append(get_num_of_punc(x['order']['word']['text'],';'))
            feat_vec.append(get_num_of_punc(x['order']['word']['text'],')'))
            feat_vec.append(get_num_of_punc(x['order']['word']['text'],'('))
            feat_vec.append(get_num_of_punc(x['order']['word']['text'],'/'))
            
            # get actual sentence
            word2vec_sent = ''
            word_vecs = []
            for word in x['order']['lemma']['text']:
                if str(word) in model:
                    word_vecs.append( [float(i) for i in model[word]] ) 
            word_vec_list.append( (np.sum(np.array(word_vecs),0)).tolist() )

            if predict:
                pre_actual_sent = ''
                for word in x['order']['word']['text']:
                    pre_actual_sent += word + ' '
                pre_actual_sent_list.append(pre_actual_sent) 
            else:
                for word in x['order']['lemma']['text']:
                    if str(word) in model: 
                        word2vec_sent += word + ' '

            # adding linguistic features
            # https://msu.edu/~jdowell/135/transw.html
            feat_vec.append(float(get_semantic_score(model,topic_dict,word2vec_sent,file_name[:5])))
            for term in sum_terms:
                feat_vec.append(contains_term(term,word2vec_sent))
            
            # get list of tf-idf scores
            tfidf = []
            for score in x['order']['word']['tfidf']:
                tfidf.append(score)
            tfidf.extend( [0.]*(200-len(tfidf)) )
            feat_vec.extend(tfidf) # temporary
            
            # append x_set, y_set
            feat_num = len(feat_vec)
            feat_vec_list.append( feat_vec )

            # update score list
            if not(predict):
                score = 0.0
                for summary in sum_dict[file_name[:5]]:
                    score += get_sim(word2vec_sent,summary,stopwords) # to get similarity score
                    doc_num = len(sum_dict[file_name[:5]])
                score = get_score_label(float(score/doc_num))
                if score<0: score = 0
                scores.append(score)

            if predict: p_feat_vec_list.append( feat_vec )
        if predict: pickle.dump( ( pre_actual_sent_list, p_feat_vec_list, pos_list ), open('predict/'+file_name,'wb'),2 )

    print('Total datasets: ' + str(len(feat_vec_list)) )
    print('Unique_scores: ' + str( len(set(scores)) ) )
    return ( feat_vec_list, scores, word_vec_list )

if __name__=="__main__":
    global feat_num
    feat_num = 0
    # 1 for predict 0 for training
    predict = True if int(sys.argv[1])==1 else False
    if predict: 
        get_data_pair(predict=predict)
    else:
        pickle.dump( get_data_pair(predict=predict), open('sum.pkl','wb'),2 )
        print( "There are " + str(feat_num) + ' features')
        
        
