"""
This code is adapted from:
http://deeplearning.net/tutorial/code/logistic_sgd.py
"""

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

def sgd_optimization_sum(learning_rate=0.13, n_epochs=10000000,
                           dataset='sum.pkl',
                           batch_size=2):
    """
    Gradient descent optimization of a log-linear
    model

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the summary dataset file from
    """

    dataset = pickle.load( open( dataset, "rb" ) )

    ###############
    # MODEL SETUP #
    ###############
    models = BuildModel( 	learning_rate=0.13, 
    						n_epochs=1000,
                			dataset='sum.pkl',
                			batch_size=batch_size )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(models.n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(models.n_train_batches):
            minibatch_avg_cost = models.train_model(minibatch_index)

            #print('train error %f' % minibatch_avg_cost)

            # iteration number
            iter = (epoch - 1) * models.n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [models.validate_model(i)
                                     for i in range(models.n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        models.n_train_batches,
                        this_validation_loss * 100.
                    )
                )
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [models.test_model(i)
                                   for i in range(models.n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            models.n_train_batches,
                            test_score * 100.
                        )
                    )
                    # save the best model
                    with open('model.pkl', 'wb') as f:
                        pickle.dump(models.classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (  \
        epoch, 1. * epoch / (end_time - start_time)))
    #print(('The code for file '+os.path.split(__file__)[1] + ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)

def predict():
    """
    load a trained model and use it to predict prob.
    """

    # load the saved model
    classifier = pickle.load(open('model.pkl'))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)

    test_set_x = [[0, 0, 2, 4, 3, 9, 0, 2, 2, 2, 3, 1, 2]]
    predicted_values = predict_model(test_set_x)
    print("Predicted values")
    print(predicted_values)

if __name__ == '__main__':
    sgd_optimization_sum()
    predict()
