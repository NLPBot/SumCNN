"""
This code is adapted from:
http://deeplearning.net/tutorial/code/logistic_sgd.py
"""

import theano
from theano import tensor as T
import numpy as np
from LogisticRegression import *

class BuildModel(object):
    def __init__(self,learning_rate=0.13, n_epochs=1000,
                           dataset='sum.pkl.gz',
                           batch_size=600):
        """
        stochastic gradient descent optimization of a log-linear model

        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
                              gradient)

        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer

        :type dataset: string
        :param dataset: the path of the summary dataset file from
        """

        datasets = self.load_data(dataset)

        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        # compute number of minibatches for training, validation and testing
        self.n_train_batches = train_set_x.get_value(borrow=True).shape[0] # batch_size
        self.n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] # batch_size
        self.n_test_batches = test_set_x.get_value(borrow=True).shape[0] # batch_size

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print('... building the model')

        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch

        # generate symbolic variables for input (x and y represent a
        # minibatch)
        x = T.matrix('x')  # data, presented as rasterized images
        y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

        # construct the logistic regression class
        # Each MNIST image has size 28*28
        self.classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)



        # cost = negative log likelihood in symbolic format
        cost = classifier.negative_log_likelihood(y)

        # compiling a Theano function that computes the mistakes that are made by
        # the model on a minibatch
        self.test_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        self.validate_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        # compute the gradient of cost with respect to theta = (W,b)
        g_W = T.grad(cost=cost, wrt=classifier.W)
        g_b = T.grad(cost=cost, wrt=classifier.b)

        ############################################# insert MLP ####################

        # start-snippet-3
        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs.
        updates = [(classifier.W, classifier.W - learning_rate * g_W),
                   (classifier.b, classifier.b - learning_rate * g_b)]

        # compiling a Theano function `train_model` that returns the cost, but in
        # the same time updates the parameter of the model based on the rules
        # defined in `updates`
        self.train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

    def load_data(self,dataset):
        ''' Loads the dataset
        :type dataset: string
        :param dataset: the path to the sum dataset 
        '''

        #############
        # LOAD DATA #
        #############

        # Load the dataset
        with gzip.open(dataset, 'rb') as f:
            try:
                train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
            except:
                train_set, valid_set, test_set = pickle.load(f)
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