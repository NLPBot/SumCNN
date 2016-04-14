import json
from pprint import pprint
import os
import pickle

def get_features():

    feat_vec_list = []
    scores = []
    score = 0.5

    data_dir = os.path.join('data','features')
    for file_name in os.listdir(data_dir):
        with open('data/features/'+file_name) as data_file:    
            data = json.load(data_file)

        # Dictionaries to read files
        """
        ner = defaultdict()
        pos = defaultdict()
        position = defaultdict()
        pronoun = defaultdict()
        ratio = defaultdict()
        """
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
            feat_vec.append(x['position']['paragraph']['forward'])
            feat_vec.append(x['position']['paragraph']['reverse'])
            feat_vec.append(x['position']['sentence']['document']['forward'])
            feat_vec.append(x['position']['sentence']['document']['reverse'])
            feat_vec.append(x['position']['sentence']['paragraph']['forward'])
            feat_vec.append(x['position']['sentence']['paragraph']['reverse'])
            # append x_set, y_set
            feat_vec_list.append( feat_vec )
            scores.append(score)

    return ( feat_vec_list, scores )

if __name__=="__main__":
    # pre-set
    feat_num = 13
    pickle.dump( get_features(), open('sum.pkl','wb') )
    
