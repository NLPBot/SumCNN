"""
This code is adapted from:
http://deeplearning.net/tutorial/code/logistic_sgd.py
"""

import theano
from theano import tensor as T
import numpy as np
import numpy
from LogisticRegression import *
import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
from HiddenLayer import *
from LeNetConvPoolLayer import *
from theano.tensor.nnet import conv2d
import pickle

class BuildModel(object):
    def __init__(self,learning_rate=0.13, n_epochs=10000000,
                           dataset='sum.pkl.gz',
                           batch_size=2):
        """
        stochastic gradient descent optimization of a log-linear model

        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
                              gradient)

        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer

        :type dataset: string
        :param dataset: the path of the summary dataset file
        """

        ######################
        #   Preparing Data   #
        ######################
        print('\n... Preparing Data')

        datasets = self.load_data(dataset,batch_size)

        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        print( 'train_set_x dimensions ' + str(train_set_x.get_value(borrow=True).shape[0]) + ' ' + 
            str(train_set_x.get_value(borrow=True).shape[1]) )
        print( 'valid_set_x dimensions ' + str(valid_set_x.get_value(borrow=True).shape[0]) + ' ' + 
            str(valid_set_x.get_value(borrow=True).shape[1]) )
        print( 'test_set_x dimensions ' + str(test_set_x.get_value(borrow=True).shape[0]) + ' ' + 
            str(test_set_x.get_value(borrow=True).shape[1]) )

        # compute number of minibatches for training, validation and testing
        self.n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
        self.n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
        self.n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

        print( 'n_train_batches ' + str(self.n_train_batches) )
        print( 'n_valid_batches ' + str(self.n_valid_batches) )
        print( 'n_test_batches ' + str(self.n_test_batches) +'\n' )

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print('... building the model')

        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch

        # generate symbolic variables for input (x and y represent a minibatch)
        x = T.matrix('x')  # data
        y = T.ivector('y')  # probs, presented as 1D vector of [int] labels

        feature_num = 13
########################################################################################
        """
        rng = numpy.random.RandomState(23455)

        feature_num = 10
        dim_1 = 10
        dim_2 = feature_num    
        nkerns=20

        # Reshape matrix of rasterized images of shape (batch_size, 1 * 10)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        # (10,) is the size of feature vectors.
        conv_layer_input = x.reshape((10, 1, dim_1, dim_2))

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (10-3+1 , 10-3+1) = (8, 8)
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
        # 4D output tensor is thus of shape (batch_size, nkerns[0], 4, 4)
        conv_layer = LeNetConvPoolLayer(
            rng,
            input=conv_layer_input,
            image_shape=(batch_size, 1, dim_1, dim_2),
            filter_shape=(nkerns, 1, 2, 2),
            poolsize=(2, 2)
        )

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, nkerns[1]*4*4),
        # or (10, 50 * 4 * 4) = (10, 800) with the default values.
        conv_layer_output = conv_layer.output.flatten(2)
        """
########################################################################################

        # Set up vars
        rng = numpy.random.RandomState(23455)
        n_in_0 = feature_num
        n_out_0 = 5000

        # construct a fully-connected sigmoidal layer
        layer0 = HiddenLayer(
            rng,
            input=x,
            n_in=n_in_0,
            n_out=n_out_0,
            activation=T.tanh
        )

        # construct a fully-connected sigmoidal layer
        n_in_1 = n_out_0
        n_out_1 = 1000
        layer1 = HiddenLayer(
            rng,
            input=layer0.output,
            n_in=n_in_1,
            n_out=n_out_1,
            activation=T.tanh
        )

        # classify the values of the fully-connected sigmoidal layer
        num_of_class = 3 # (sim > 0.75 => 2), (sim < 0.25 => 0), ( 0.25<sim<0.75 )        
        self.classifier = LogisticRegression(input=layer1.output, n_in=n_out_1, n_out=num_of_class)

        # cost = negative log likelihood in symbolic format
        cost = self.classifier.negative_log_likelihood(y)

        # batch_size == row size == weight vector row size
        self.test_model = theano.function(
            inputs=[index],
            outputs=self.classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        self.validate_model = theano.function(
            inputs=[index],
            outputs=self.classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        # create a update list by gradient descent
        params = layer1.params + layer0.params + [self.classifier.W, self.classifier.b] # + conv_layer.params 
        grads = T.grad(cost, params)
        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)
        ]

        # train model
        self.train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

    def load_word2vec(self):
        print "loading data...",
        x = cPickle.load(open("mr.p","rb"))
        self.revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
        print "data loaded!"
        

    def load_data(self,dataset,batch_size):
        ''' Loads the dataset
        :type dataset: string
        :param dataset: the path to the sum dataset 
        '''

        #############
        # LOAD DATA #
        #############

        data = pickle.load( open( dataset, "rb" ) )
        train_set = data
        valid_set = data
        test_set = data

        def shared_dataset(data_xy, borrow=True):
            """ Function that loads the dataset into shared variables
            """
            data_x, data_y = data_xy
            shared_x = theano.shared(numpy.asarray(data_x,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            shared_y = theano.shared(numpy.asarray(data_y,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)

            return shared_x, T.cast(shared_y, 'int32')

        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                (test_set_x, test_set_y)]
        return rval