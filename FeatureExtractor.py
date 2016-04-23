import json
from pprint import pprint
import os, pickle, sys
from SentSim import *
import nltk
from nltk.corpus import stopwords
from som import *
import gensim
import numpy as np
import xml.etree.ElementTree as ET
from ngram import *
import string

def load_word2vec():
    return gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)

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

def get_data_pair(predict=False):
    global feat_num

    # setting up
    feat_vec_list, scores = [], []
    score = 0
    if not(predict): 
        model = load_word2vec()
        sum_dict = read_gold_sum(model)
        topic_dict = get_topics()
    stopwords = nltk.corpus.stopwords.words('english')

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
            if predict:
                pre_actual_sent = ''
                for word in x['order']['word']['text']:
                    if str(word) not in stopwords:
                        pre_actual_sent += word + ' '
                pre_actual_sent_list.append(pre_actual_sent) 
            else:
                for word in x['order']['lemma']['text']:
                    if str(word) in model and str(word) not in stopwords: 
                        word2vec_sent += word + ' '

            feat_vec.append(float(get_semantic_score(model,topic_dict,word2vec_sent,file_name[:5])))

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
    return ( feat_vec_list, scores )

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
        
        
