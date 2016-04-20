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

def predict(file_name, actual_sent_list, pos_list, test_set_x):
    """
    load a trained model and use it to predict prob.
    """

    # load the saved model
    classifier = pickle.load(open('model.pkl'))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)

    print("Predicting..."+file_name)
    predicted_values = predict_model(test_set_x)
    result = zip(actual_sent_list,predicted_values,pos_list)
    print('Scores: '+str(predicted_values) )
    pickle.dump( result, open('result/'+file_name+'.pkl','wb') )

# return lists of ( actual_sent, score, position )
def read_result(file_name):
    return pickle.load( open('result/'+file_name+'.pkl', "rb") )

