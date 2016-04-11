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

class BuildModel(object):
    def __init__(self,learning_rate=0.13, n_epochs=10000000,
                           dataset='sum.pkl.gz',
                           batch_size=10):
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

        # Set up vars
        rng = numpy.random.RandomState(23455)
        n_in = 10
        n_out = 10
        print( 'n_in ' + str(n_in) )
        print( 'n_out ' + str(n_out) + '\n' )

        # construct a fully-connected sigmoidal layer
        layer0 = HiddenLayer(
            rng,
            input=x,
            n_in=n_in,
            n_out=n_out,
            activation=T.tanh
        )

        # construct a fully-connected sigmoidal layer
        layer1 = HiddenLayer(
            rng,
            input=layer0.output,
            n_in=n_out,
            n_out=n_in,
            activation=T.tanh
        )

        # classify the values of the fully-connected sigmoidal layer        
        self.classifier = LogisticRegression(input=layer1.output, n_in=n_in, n_out=n_out)

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
        params = layer1.params + layer0.params + [self.classifier.W, self.classifier.b]
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

    def load_data(self,dataset,batch_size):
        ''' Loads the dataset
        :type dataset: string
        :param dataset: the path to the sum dataset 
        '''

        #############
        # LOAD DATA #
        #############

        # Download the MNIST dataset if it is not present
        data_dir, data_file = os.path.split(dataset)
        if data_dir == "" and not os.path.isfile(dataset):
            # Check if dataset is in the data directory.
            new_path = os.path.join(
                os.path.split(__file__)[0],
                "data",
                dataset
            )
            if os.path.isfile(new_path) or data_file == 'sum.pkl.gz':
                dataset = new_path

        # Load the dataset
        with gzip.open(dataset, 'rb') as f:
            try:
                train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
            except:
                train_set, valid_set, test_set = pickle.load(f)

        row = batch_size
        col = 10
        left = np.random.uniform(0,1,(row,col))
        right = np.random.uniform(0,1,(col))
        train_set = ( left, right )

        left = np.random.uniform(0,1,(row,col))
        right = np.random.uniform(0,1,(col))
        valid_set = ( left, right )

        left = np.random.uniform(0,1,(row,col))
        right = np.random.uniform(0,1,(col))
        test_set = ( left, right )

        # train_set, valid_set, test_set format: tuple(input, target)
        # input is a numpy.ndarray of 2 dimensions (a matrix)
        # where each row corresponds to a sentence vector. target is a
        # numpy.ndarray of 1 dimension (vector) that has the same length as
        # the number of rows in the input. It should give the target
        # to the sentence vector with the same index in the input.

        def shared_dataset(data_xy, borrow=True):
            """ Function that loads the dataset into shared variables

            The reason we store our dataset in shared variables is to allow
            Theano to copy it into the GPU memory (when code is run on GPU).
            Since copying data into the GPU is slow, copying a minibatch everytime
            is needed (the default behaviour if the data is not in a shared
            variable) would lead to a large decrease in performance.
            """
            data_x, data_y = data_xy
            shared_x = theano.shared(numpy.asarray(data_x,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            shared_y = theano.shared(numpy.asarray(data_y,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            # When storing data on the GPU it has to be stored as floats
            # therefore we will store the labels as ``floatX`` as well
            # (``shared_y`` does exactly that). But during our computations
            # we need them as ints (we use labels as index, and if they are
            # floats it doesn't make sense) therefore instead of returning
            # ``shared_y`` we will have to cast it to int. This little hack
            # lets ous get around this issue
            return shared_x, T.cast(shared_y, 'int32')

        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                (test_set_x, test_set_y)]
        return rval