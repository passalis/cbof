import numpy as np
import theano
import theano.tensor as T
import lasagne
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances


class CBoF_Layer(lasagne.layers.Layer):
    """
    Lasagne implementation of the CBoF Pooling Layer
    """

    def __init__(self, incoming, n_codewords=24, V=lasagne.init.Normal(0.1), gamma=lasagne.init.Constant(0.1),
                 eps=0.00001, input_var=None, initializers=None, spatial_level=1, **kwargs):
        """
        Creates a BoF layer

        :param incoming: 
        :param n_codewords: number of codewords
        :param V: initializer used for the codebook
        :param gamma: initializer used for the scaling factors
        :param eps: epsilon used to ensure numerical stability
        :param input_var: input_var of the model (used to compile a function that extract the features fed to layer)
        :param initializers: 
        :param spatial_level: 0 (no spatial segmentation), 1 (first spatial level)
        :param pooling_type: either 'mean' or 'max'
        :param kwargs: 
        """
        super(CBoF_Layer, self).__init__(incoming, **kwargs)

        self.n_codewords = n_codewords
        self.spatial_level = spatial_level
        n_filters = self.input_shape[1]
        self.eps = eps

        # Create parameters
        self.V = self.add_param(V, (n_codewords, n_filters, 1, 1), name='V')
        self.gamma = self.add_param(gamma, (1, n_codewords, 1, 1), name='gamma')

        # Make gammas broadcastable
        self.gamma = T.addbroadcast(self.gamma, 0, 2, 3)

        # Compile function used for feature extraction
        if input_var is not None:
            self.features_fn = theano.function([input_var], lasagne.layers.get_output(incoming, deterministic=True))

        if initializers is not None:
            initializers.append(self.initialize_layer)

    def get_output_for(self, input, **kwargs):
        distances = conv_pairwise_distance(input, self.V)
        similarities = T.exp(-distances / T.abs_(self.gamma))
        norm = T.sum(similarities, 1).reshape((similarities.shape[0], 1, similarities.shape[2], similarities.shape[3]))
        membership = similarities / (norm + self.eps)

        histogram = T.mean(membership, axis=(2, 3))
        if self.spatial_level == 1:
            pivot1, pivot2 = membership.shape[2] / 2, membership.shape[3] / 2
            h1 = T.mean(membership[:, :, :pivot1, :pivot2], axis=(2, 3))
            h2 = T.mean(membership[:, :, :pivot1, pivot2:], axis=(2, 3))
            h3 = T.mean(membership[:, :, pivot1:, :pivot2], axis=(2, 3))
            h4 = T.mean(membership[:, :, pivot1:, pivot2:], axis=(2, 3))
            # Pyramid is not used in the paper
            # histogram = T.horizontal_stack(h1, h2, h3, h4)
            histogram = T.horizontal_stack(histogram, h1, h2, h3, h4)
        return histogram

    def get_output_shape_for(self, input_shape):
        if self.spatial_level == 1:
            return (input_shape[0], 5 * self.n_codewords)
        return (input_shape[0], self.n_codewords)

    def initialize_layer(self, data, n_samples=10000):
        """
        Initializes the layer using k-means (sigma is set to the mean pairwise distance)
        :param data: data
        :param n_samples: n_samples to keep for initializing the model
        :return:
        """
        if self.features_fn is None:
            assert False

        idx = np.arange(data.shape[0])
        np.random.shuffle(idx)

        features = []
        for i in range(idx.shape[0]):
            feats = self.features_fn([data[idx[i]]])
            feats = feats.transpose((0, 2, 3, 1))
            feats = feats.reshape((-1, feats.shape[-1]))
            features.extend(feats)
            if len(features) > n_samples:
                break
        features = np.asarray(features)

        kmeans = KMeans(n_clusters=self.n_codewords, n_jobs=4, n_init=5)
        kmeans.fit(features)
        V = kmeans.cluster_centers_.copy()

        # Initialize gamma
        mean_distance = np.sum(pairwise_distances(V)) / (self.n_codewords * (self.n_codewords - 1))
        self.gamma.set_value(self.gamma.get_value() * np.float32(np.sqrt(1) * mean_distance))

        # Initialize codebook
        V = V.reshape((V.shape[0], V.shape[1], 1, 1))
        self.V.set_value(np.float32(V))


def conv_pairwise_distance(feature_maps, codebook):
    """
    Calculates the pairwise distances between the feature maps (n_samples, filters, x, y)
    :param feature_maps: 
    :param codebook: 
    :return: 
    """
    x_square = T.sum(feature_maps ** 2, axis=1)  # n_samples, filters, x, y
    x_square = x_square.reshape((x_square.shape[0], 1, x_square.shape[1], x_square.shape[2]))
    x_square = T.addbroadcast(x_square, 1)

    y_square = T.sum(codebook ** 2, axis=1)
    y_square = y_square.reshape((1, y_square.shape[0], y_square.shape[1], y_square.shape[2]))
    y_square = T.addbroadcast(y_square, 0, 2, 3)

    inner_product = T.nnet.conv2d(feature_maps, codebook)
    dist = x_square + y_square - 2 * inner_product
    dist = T.sqrt(T.maximum(dist, 0))
    return dist
