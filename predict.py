import theano
from theano import tensor as T
import numpy as np
from BuildModel import *
import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
import pickle
from ngram import *

def predict(file_name):
    """
    load a trained model and use it to predict prob.
    """
    # load the saved model
    classifier = pickle.load(open('model.pkl'))
    data_xy = pickle.load( open( 'predict/'+file_name, "rb" ) )
    actual_sent_list, data_x, pos_list = data_xy

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)

    print("Predicting..."+file_name)
    predicted_values = predict_model(data_x)
    result = zip(actual_sent_list,predicted_values,pos_list)
    result = sorted(result, key=lambda x: x[1], reverse=True)
    #print(result)
    print('Scores: '+str(predicted_values) )
    pickle.dump( result, open('result/'+file_name[:30]+'.pkl','wb') )

# return lists of ( actual_sent, score, position )
def read_result(file_name):
    return pickle.load( open('result/'+file_name+'.pkl', "rb") )

if __name__ == '__main__':
    sub_dirs = ['feat_docs_para','feat_docs_sent','feat_model_para','feat_model_sent']
    data_dir = os.path.join('data',sub_dirs[1])
    for file_name in os.listdir(data_dir):
        predict(file_name)
        

        
        
        
