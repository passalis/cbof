import lasagne
import theano
import theano.tensor as T
from model.nbof import CBoF_Input_Layer
from model.base_learner import Base_Learner
from model.cnn_feat import CNN_Feature_Extractor


class CBoF(Base_Learner):
    def __init__(self, n_classes=10, learning_rate=0.00001, bof_layer=(4, 0, 128), hidden_neurons=(1000,),
                 dropout=(0.5,), feature_dropout=0, g=0.1):

        Base_Learner.__init__(self)

        input_var = T.ftensor4('inputs')
        target_var = T.ivector('targets')

        # Create the CNN feature extractor
        self.cnn_layer = CNN_Feature_Extractor(input_var, size=None)

        # Create the BoF layer
        (cnn_layer_id, spatial_level, n_codewords) = bof_layer
        self.bof_layer = CBoF_Input_Layer(input_var, self.cnn_layer, cnn_layer_id, level=spatial_level,
                                          n_codewords=n_codewords, g=g, pyramid=False)
        features = self.bof_layer.fused_features
        n_size_features = self.bof_layer.features_size

        # Create an output MLP
        network = lasagne.layers.InputLayer(shape=(None, n_size_features), input_var=features)
        if feature_dropout > 0:
            network = lasagne.layers.DropoutLayer(network, p=feature_dropout)
        for n, drop_rate in zip(hidden_neurons, dropout):
            network = lasagne.layers.DenseLayer(network, num_units=n, nonlinearity=lasagne.nonlinearities.elu,
                                                W=lasagne.init.Orthogonal())
            network = lasagne.layers.DropoutLayer(network, p=drop_rate)

        network = lasagne.layers.DenseLayer(network, num_units=n_classes,
                                            nonlinearity=lasagne.nonlinearities.softmax,
                                                W=lasagne.init.Normal(std=1))
        # Get network loss
        self.prediction_train = lasagne.layers.get_output(network, deterministic=False)
        loss = lasagne.objectives.categorical_crossentropy(self.prediction_train, target_var).mean()

        # Define training rules
        params_mlp = lasagne.layers.get_all_params(network, trainable=True)
        updates_mlp = lasagne.updates.adam(loss, params_mlp, learning_rate=learning_rate)
        updates = lasagne.updates.adam(loss, params_mlp, learning_rate=learning_rate)
        updates.update(lasagne.updates.adam(loss, self.cnn_layer.layer_params[cnn_layer_id],
                                            learning_rate=learning_rate))
        updates.update(lasagne.updates.adam(loss, self.bof_layer.V, learning_rate=learning_rate))
        updates.update(lasagne.updates.adam(loss, self.bof_layer.sigma, learning_rate=learning_rate))

        # Define testing/validation
        prediction_test = lasagne.layers.get_output(network, deterministic=True)

        # Compile functions
        self.train_fn = theano.function([input_var, target_var], loss, updates=updates)
        self.train_mlp_fn = theano.function([input_var, target_var], loss, updates=updates_mlp)
        self.test_fn = theano.function([input_var], T.argmax(prediction_test, axis=1))

        # Get the output of the bof module
        self.get_features_fn = theano.function([input_var], features)

    def init_bof(self, data):
        """
        Initializes the BoF layer using k-means
        :param data:
        :return:
        """
        self.bof_layer.initialize(data)
