import lasagne
import lasagne.layers.dnn
import theano
import theano.tensor as T
from model.cnn_feat import CNN_Feature_Extractor
from model.base_learner import Base_Learner


class CNN_Simple(Base_Learner):
    """
    Implements the baseline models (CNN and SPP)
    """

    def __init__(self, learning_rate=0.0001, hidden_neurons=(1000,), dropout=(0.5,),  feature_dropout=0.5, n_classes=15,
                 use_spatial_pooling=False, pool_dims=[2, 1], size=28):

        Base_Learner.__init__(self)

        input_var = T.ftensor4('inputs')
        target_var = T.ivector('targets')

        if use_spatial_pooling:
            size = None

        # Create the CNN feature extractor
        self.cnn_layer = CNN_Feature_Extractor(input_var, size=size, pool_size=[(2, 2), ()])
        network = self.cnn_layer.networks[-1]
        cnn_params = self.cnn_layer.layer_params[-1]

        # Add spatial pooling layer, if needed
        if use_spatial_pooling:
            # network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=(1,1),
            #                                      nonlinearity=lasagne.nonlinearities.rectify,
            #                                      W=lasagne.init.GlorotUniform())
            network = lasagne.layers.dnn.SpatialPyramidPoolingDNNLayer(network, pool_dims=pool_dims)
        else:
            # otherwise, add a regular 2x2 pooling layer
            network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

        if feature_dropout > 0:
            network = lasagne.layers.DropoutLayer(network, p=feature_dropout)

        params_mlp = []
        for n, drop_rate in zip(hidden_neurons, dropout):
            network = lasagne.layers.DenseLayer(network, num_units=n, nonlinearity=lasagne.nonlinearities.elu,
                                                W=lasagne.init.Orthogonal())
            params_mlp.append(network.W)
            params_mlp.append(network.b)
            network = lasagne.layers.DropoutLayer(network, p=drop_rate)

        network = lasagne.layers.DenseLayer(network, num_units=n_classes,
                                            nonlinearity=lasagne.nonlinearities.softmax)
        params_mlp.append(network.W)
        params_mlp.append(network.b)

        # Get network loss
        prediction_train = lasagne.layers.get_output(network, deterministic=False)
        loss = lasagne.objectives.categorical_crossentropy(prediction_train, target_var).mean()

        # Define training rules
        updates_mlp = lasagne.updates.adam(loss, params_mlp, learning_rate=learning_rate)
        updates = lasagne.updates.adam(loss, params_mlp, learning_rate=learning_rate)
        updates.update(lasagne.updates.adam(loss, cnn_params, learning_rate=learning_rate))

        # Define testing/validation
        prediction_test = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(prediction_test, target_var).mean()
        test_acc = T.mean(T.eq(T.argmax(prediction_test, axis=1), target_var), dtype='float32')

        # Compile functions
        self.train_fn = theano.function([input_var, target_var], loss, updates=updates)
        self.test_fn = theano.function([input_var], T.argmax(prediction_test, axis=1))
        self.val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

        self.train_mlp_fn = theano.function([input_var, target_var], loss, updates=updates_mlp)
