import lasagne
import theano
import theano.tensor as T
import numpy as np
from datasets.mnist import load_mnist
from models.bof import CBoF_Layer
from models.learner_base import LearnerBase


class LeNeT_Model(LearnerBase):
    def __init__(self, pooling='spp', spatial_level=1, n_codewords=64, learning_rate=0.001):
        self.initializers = []

        input_var = T.ftensor4('input_var')
        target_var = T.ivector('targets')

        network = lasagne.layers.InputLayer(shape=(None, 1, None, None), input_var=input_var)
        network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(5, 5),
                                             nonlinearity=lasagne.nonlinearities.rectify)
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
        network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=(5, 5),
                                             nonlinearity=lasagne.nonlinearities.rectify)
        if pooling == 'spp':
            network = lasagne.layers.SpatialPyramidPoolingLayer(network, pool_dims=[1, 2])
        elif pooling == 'bof':
            network = CBoF_Layer(network, input_var=input_var, initializers=self.initializers, n_codewords=n_codewords,
                                 spatial_level=spatial_level)

        network = lasagne.layers.dropout(network, p=.5)
        network = lasagne.layers.DenseLayer(network, num_units=1000, nonlinearity=lasagne.nonlinearities.elu)
        network = lasagne.layers.dropout(network, p=.5)
        network = lasagne.layers.DenseLayer(network, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)
        self.network = network

        train_prediction = lasagne.layers.get_output(network, deterministic=False)
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        loss = lasagne.objectives.categorical_crossentropy(train_prediction, target_var).mean()

        self.params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.adam(loss, self.params, learning_rate=learning_rate)

        self.train_fn = theano.function([input_var, target_var], loss, updates=updates)
        self.test_fn = theano.function([input_var], T.argmax(test_prediction, axis=1))

        print "Model Compiled!"

    def initialize_model(self, data, n_samples=50000):
        for initializer in self.initializers:
            initializer(data, n_samples=n_samples)
            print "Model initialized!"


if __name__ == '__main__':
    np.random.seed(12345)

    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()

    for pool_type in ['bof', 'spp']:
        model = LeNeT_Model(pooling=pool_type)

        if pool_type == 'bof':
            model.initialize_model(X_train, n_samples=50000)

        model.train_model(X_train, y_train, validation_data=X_val, validation_labels=y_val, n_iters=50, batch_size=256)

        print "Evaluated model = ", pool_type
        print "Error = ", (1 - model.test_model(X_test, y_test)) * 100
        print "Error (0.7 scale) = ", (1 - model.test_model(X_test, y_test, scale=0.7)) * 100
        print "Error (0.8 scale) = ", (1 - model.test_model(X_test, y_test, scale=0.8)) * 100
