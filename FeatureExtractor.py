import json
from pprint import pprint
import os, cPickle
import pickle
from SentSim import *
import nltk
from nltk.corpus import stopwords
from predict import *

def load_word2vec():
    print "loading data...",
    x = cPickle.load(open("word2vec.pkl","rb"))
    word2vec = x[0]
    print "data loaded!"
    return word2vec
    
def get_sim(sent,summary):
    ratio = float(len(summary.split()))/float(len(sent.split()))
    return (binary_similarity(sent,summary)+frequency_similarity(sent,summary))/2

def read_gold_sum():
    sum_dict = {}
    gold_sum = ''
    data_dir = os.path.join('data','gold_sum')

    print 'reading gold summaries...',
    for file_name in os.listdir(data_dir):
        with open('data/gold_sum/'+file_name) as data_file: 
			gold_sum = data_file.read()
        if file_name[:5] not in sum_dict.keys(): 
			sum_dict[file_name[:5]] = [ gold_sum ]
        else:
			sum_dict[file_name[:5]].append( gold_sum )
    print 'Done reading gold summaries...'
    return sum_dict

def get_ratio(e,total):
    return float(e)/float(total)

def get_score_label(score):
    return int(score*100)

def get_data_pair(training=True):

    feat_vec_list = []
    feat_vec_list_predict = []
    scores = []
    score = 0
    if training:
        sum_dict = read_gold_sum()
    #word2vec = load_word2vec()
    #stopwords = nltk.corpus.stopwords.words('english')
    sub_dirs = ['feat_docs_para','feat_docs_sent','feat_model_para','feat_model_sent']

    data_dir = os.path.join('data',sub_dirs[1])
    score_index = 0
    for file_name in os.listdir(data_dir):
        with open('data/'+sub_dirs[1]+'/'+file_name) as data_file:    
            data = json.load(data_file)

        actual_sent_list = []
        pos_list = []
        feat_vec_list_predict = []
        for x in data['data']['documents']['document']:
            # append feature list
            feat_vec = []

            # 12 features
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

            # get actual sentence
            actual_sent = ''
            #wordvec = numpy.array([0.]*300)
            for word in x['order']['word']['text']:
                #if str(word) in word2vec.keys() and str(word) not in stopwords: 
                #    wordvec += word2vec[word]
                actual_sent += word + ' '
            actual_sent_list.append(actual_sent)

            #wordvec = wordvec / len(x['order']['word']['text'])
            #feat_vec.extend(wordvec)

            # 30 features
            # get list of tf-idf scores
            tfidf = []
            for score in x['order']['word']['tfidf']:
                tfidf.append(score)
            tfidf.extend( [0.]*(100-len(tfidf)) )
            feat_vec.append(max(tfidf)) # temporary
            
            # append x_set, y_set
            feat_vec_list.append( feat_vec )
            if not(training):
                feat_vec_list_predict.append( feat_vec )

            # update score list
            if training:
                score = get_sim(actual_sent,sum_dict[file_name[:5]][0]) # to get similarity score
                scores.append(get_score_label(score))
        if not(training):
            predict(file_name, actual_sent_list, pos_list, feat_vec_list_predict)

    print('Total datasets: ' + str(len(feat_vec_list)) )
    return ( feat_vec_list, scores )

if __name__=="__main__":
    # pre-set
    feat_num = 13
    pickle.dump( get_data_pair(training=False), open('sum.pkl','wb') )

