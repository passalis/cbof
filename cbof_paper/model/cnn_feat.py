import lasagne

class Base_CNN_Feature_Extractor:
    def __init__(self):
        # Features extracted from each layer
        self.layer_features = []

        # Feature dimension per layer
        self.features_dim = []

        # Cumulative parameters for each layer
        self.layer_params = []

        # Lasagne network reference for each layer
        self.networks = []

    def get_features(self, layer):
        """
        Returns all the feature vectors of a layer
        :param layer: the layer from which to extract the feature vectors
        :return: the feature vectors
        """

        features = self.layer_features[layer]
        features = features.reshape((features.shape[0], features.shape[1] * features.shape[2], features.shape[3]))
        return features

    def get_spatial_features(self, layer, i, level=1):
        """
        Returns the features of the i-th region of the layer (only 2x2 segmentation is supported)
        :param layer: the layer from which to extract the features
        :param i: the region of the layer to extract the features
        :return: the feature vectors
        """
        # This function assumes a square image input
        pivot = self.layer_features[layer].shape[1] // 2
        if level == 1:
            if i == 0:
                features = self.layer_features[layer][:, :pivot, :pivot, :]
            elif i == 1:
                features = self.layer_features[layer][:, :pivot, pivot:, :]
            elif i == 2:
                features = self.layer_features[layer][:, pivot:, :pivot, :]
            elif i == 3:
                features = self.layer_features[layer][:, pivot:, pivot:, :]
            else:
                print("Wrong region number")
                assert False
        else:
            print("Only spatial levels 1 and 2 are supported, got ", level)
            assert False

        features = features.reshape((features.shape[0], features.shape[1] * features.shape[2], features.shape[3]))
        return features


class CNN_Feature_Extractor(Base_CNN_Feature_Extractor):
    """
    Implements a simple convolutional feature extractor
    """

    def __init__(self, input_var, size=28, channels=1, n_filters=[32, 64], filters_size=[(5, 5), (5, 5)],
                 pool_size=[(2, 2), ()]):
        """
        Defines a set of convolutional layer that extracts features
        :param input_var: input var of the network
        :param size: image input size (set to None to allow images of arbitrary size)
        :param channels: number of channels in the image
        :param n_filters: number of filters in each layer
        :param filters_size: size of the filters in each layer
        :param pool_size: pool size in each layer
        """
        # Input
        network = lasagne.layers.InputLayer(shape=(None, channels, size, size), input_var=input_var)

        # Store the dimensionality of each feature vector (for use by the next layers)
        self.features_dim = []
        # Store the features of each convolutional layer
        self.layer_features = []
        self.layer_params = []
        self.networks = []

        # Define the layers
        for n, size, pool in zip(n_filters, filters_size, pool_size):
            network = lasagne.layers.Conv2DLayer(network, num_filters=n, filter_size=size,
                                                 nonlinearity=lasagne.nonlinearities.rectify,
                                                 W=lasagne.init.GlorotUniform())
            self.features_dim.append(n)
            if pool:
                network = lasagne.layers.MaxPool2DLayer(network, pool_size=pool)

            # Save the output of each layer (after reordering the dimensions: n_samples, n_vectors, n_feats)
            self.layer_features.append(lasagne.layers.get_output(network, deterministic=True).transpose((0, 2, 3, 1)))
            self.layer_params.append(lasagne.layers.get_all_params(network, trainable=True))
            self.networks.append(network)
