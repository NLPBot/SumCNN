import json
from pprint import pprint
import os, cPickle
import pickle
from SentSim import *
import nltk
from nltk.corpus import stopwords

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
    stopwords = nltk.corpus.stopwords.words('english')
    word2vec = load_word2vec()

    print('reading gold summaries...')
    for file_name in os.listdir(data_dir):
        with open('data/gold_sum/'+file_name) as data_file: 
			gold_sum = data_file.read()
        vec_concat = [0.]*300
        for term in gold_sum.split():
            if term in stopwords: continue
            if term in word2vec.keys(): 
                vec_concat = vec_concat + word2vec[term] 
            # to get word vector mean of all words within sentence
            vec_concat = vec_concat / len(gold_sum.split())

        if file_name[:5] not in sum_dict.keys(): 
			sum_dict[file_name[:5]] = [ (gold_sum,vec_concat) ]
        else:
			sum_dict[file_name[:5]].append( (gold_sum,vec_concat) )
    print('Done reading gold summaries...')
    return sum_dict, word2vec

def get_sim_scores():
    score_list = []
    sub_dirs = ['feat_docs_para','feat_docs_sent','feat_model_para','feat_model_sent']
    data_dir = os.path.join('data',sub_dirs[1])
    sum_dict, word2vec = read_gold_sum()
    stopwords = stopwords.words('english')

    for file_name in os.listdir(data_dir):
        with open('data/'+sub_dirs[1]+'/'+file_name) as data_file:    
            data = json.load(data_file)
        # Dictionaries to read files
        scores = []
        for x in data['data']['documents']['document']:
            temp_sent = ''
            vec_concat = [0.]*300
            for term in x['term']:
                if term in word2vec.keys() and term not in stopwords:
                    vec_concat = vec_concat + word2vec[term] 
                temp_sent += term + ' '     
            # to get word vector mean of all words within sentence
            vec_concat = vec_concat / len(x['term'])
            # to get similarity score
            score = get_sim(temp_sent,sum_dict[file_name[:5]][0][0])
            score += cos_sim(sum_dict[file_name[:5]][1],vec_concat)
            scores.append(score)
        # normalize
        score_list.extend( [ float(score/max(scores)) for score in scores ] )
    return score_list

def get_ratio(e,total):
    return float(e)/float(total)

def get_score_label(score):
    label = 0
    if score<0.10:
        label = 0
    if score<0.20 and score>0.10:
        label = 1
    if score<0.30 and score>0.20:
        label = 2
    if score<0.40 and score>0.30:
        label = 3
    if score<0.50 and score>0.40:
        label = 4
    if score<0.60 and score>0.50:
        label = 5
    if score<0.70 and score>0.60:
        label = 6
    if score<0.80 and score>0.70:
        label = 7
    if score<0.90 and score>0.80:
        label = 8
    if score>0.90:
        label = 9  
    return label

def get_tf_idf_list(data):
    for x in data['data']['terms']:
        print(x)

def get_data_pair():

    feat_vec_list = []
    scores = []
    score = 0

    sum_dict = read_gold_sum()
    score_list = get_sim_scores()
    sub_dirs = ['feat_docs_para','feat_docs_sent','feat_model_para','feat_model_sent']

    data_dir = os.path.join('data',sub_dirs[1])
    score_index = 0
    for file_name in os.listdir(data_dir):

        with open('data/'+sub_dirs[1]+'/'+file_name) as data_file:    
            data = json.load(data_file)

        # Dictionaries to read files
        """
        ner = defaultdict()
        pos = defaultdict()
        position = defaultdict()
        pronoun = defaultdict()
        ratio = defaultdict()
        """

        get_tf_idf_list(data)

        for x in data['data']['documents']['document']:
            # append feature list
            feat_vec = []
            feat_vec.append(x['conj']['+'])
            feat_vec.append(x['conj']['-'])
            feat_vec.append(x['count']['punct'])
            feat_vec.append(x['count']['stop0'])
            feat_vec.append(x['count']['stop1'])
            feat_vec.append(x['length'])
            feat_vec.append(x['overlap'])

            total = float(x['position']['paragraph']['forward']) + float(x['position']['paragraph']['reverse'])            
            feat_vec.append(get_ratio(float(x['position']['paragraph']['forward']),total))

            total = float(x['position']['sentence']['document']['forward']) + float(x['position']['sentence']['document']['reverse'])
            feat_vec.append(get_ratio(float(x['position']['sentence']['document']['forward']),total))

            total = float(x['position']['sentence']['paragraph']['forward']) + float(x['position']['sentence']['paragraph']['reverse'])
            feat_vec.append(get_ratio(float(x['position']['sentence']['paragraph']['forward']),total))

            feat_vec.append(x['ratio']['punct']['word'][0])
            feat_vec.append(x['ratio']['punct']['word'][1])

            # append x_set, y_set
            feat_vec_list.append( feat_vec )
            
            temp_sent = ''
            for term in x['term']:
                temp_sent += term + ' '

            # update score list
            score = score_list[score_index]
            #print(get_score_label(score))
            scores.append(get_score_label(score))
            score_index += 1

    print('Total datasets: ' + str(len(feat_vec_list)) )
    return ( feat_vec_list, scores )

if __name__=="__main__":
    # pre-set
    feat_num = 13
    pickle.dump( get_data_pair(), open('sum.pkl','wb') )

