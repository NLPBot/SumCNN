"""
This code is adapted from:
http://deeplearning.net/tutorial/code/logistic_sgd.py
"""

import theano
from theano import tensor as T
import numpy as np
import numpy

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """ 
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        rng = np.random.RandomState(3435)
        self.W0 = theano.shared(
            value=numpy.asarray(0.01 * rng.standard_normal(size=(n_in, n_out)), dtype=theano.config.floatX),
            name='W0',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b0 = theano.shared(
            value=numpy.zeros((n_out,), dtype=theano.config.floatX),
            name='b0',
            borrow=True
        )

        n_in_1 = 10
        n_out_1 = 10

        self.W1 = theano.shared(
            value=numpy.asarray(0.01 * rng.standard_normal(size=(n_in_1, n_out_1)), dtype=theano.config.floatX),
            name='W1',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b1 = theano.shared(
            value=numpy.zeros((n_out,), dtype=theano.config.floatX),
            name='b1',
            borrow=True
        )


        layer0 = T.nnet.softmax(T.dot(input, self.W0) + self.b0)

        self.p_y_given_x = T.nnet.softmax(T.dot(layer0, self.W1) + self.b1)

        # symbolic description of prob prediction
        self.y_pred = T.nnet.softmax(T.dot(layer0, self.W1) + self.b1)

        # parameters of the model
        self.params = [self.W0, self.b0, self.W1, self.b1]

        # keep track of model input
        self.input = input

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        return T.mean(T.sub( self.p_y_given_x , y ))




