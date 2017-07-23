import theano
import numpy as np
from sklearn.preprocessing import normalize as feature_normalizer
import theano.tensor as T
import theano.gradient
import sklearn.cluster as cluster

floatX = theano.config.floatX


class NBoFInputLayer:
    """
    Defines a Neural BoF input layer
    """

    def __init__(self, g=0.1, feature_dimension=89, n_codewords=16):
        """
        Initializes the Neural BoF object
        :param g: defines the softness of the quantization
        :param feature_dimension: dimension of the feature vectors
        :param n_codewords: number of codewords / RBF neurons to be used
        """

        self.Nk = n_codewords
        self.D = feature_dimension

        # RBF-centers / codewords
        V = np.random.rand(self.Nk, self.D)
        self.V = theano.shared(value=V.astype(dtype=floatX), name='V', borrow=True)
        sigma = np.ones((self.Nk,)) / g
        self.sigma = theano.shared(value=sigma.astype(dtype=floatX), name='sigma', borrow=True)
        self.params = [self.V, self.sigma]

        # Tensor of input objects (n_objects, n_features, self.D)
        self.X = T.tensor3(name='X', dtype=floatX)

        # Feature matrix of an object (n_features, self.D)
        self.x = T.matrix(name='x', dtype=floatX)

        # Encode a set of objects
        """
        Note that the number of features per object is fixed and same for all objects.
        The code can be easily extended by defining a feature vector mask, allowing for a variable number of feature
        vectors for each object (or alternatively separately encoding each object).
        """
        self.encode_objects_theano = theano.function(inputs=[self.X], outputs=self.sym_histograms(self.X))

        # Encodes only one object with an arbitrary number of features
        self.encode_object_theano = theano.function(inputs=[self.x], outputs=self.sym_histogram(self.x))

    def sym_histogram(self, X):
        """
        Computes a soft-quantized histogram of a set of feature vectors (X is a matrix).
        :param X: matrix of feature vectors
        :return:
        """
        distances = symbolic_distance_matrix(X, self.V)
        membership = T.nnet.softmax(-distances * self.sigma)
        histogram = T.mean(membership, axis=0)
        return histogram

    def sym_histograms(self, X):
        """
        Encodes a set of objects (X is a tensor3)
        :param X: tensor3 containing the feature vectors for each object
        :return:
        """
        histograms, updates = theano.map(self.sym_histogram, X)
        return histograms

    def initialize_dictionary(self, X, max_iter=100, redo=5, n_samples=50000, normalize=False):
        """
        Samples some feature vectors from X and learns an initial dictionary
        :param X: list of objects
        :param max_iter: maximum k-means iters
        :param redo: number of times to repeat k-means clustering
        :param n_samples: number of feature vectors to sample from the objects
        :param normalize: use l_2 norm normalization for the feature vectors
        """

        # Sample only a small number of feature vectors from each object
        samples_per_object = int(np.ceil(n_samples / len(X)))

        features = None
        print("Sampling feature vectors...")
        for i in (range(len(X))):
            idx = np.random.permutation(X[i].shape[0])[:samples_per_object + 1]
            cur_features = X[i][idx, :]
            if features is None:
                features = cur_features
            else:
                features = np.vstack((features, cur_features))

        print("Clustering feature vectors...")
        features = np.float64(features)
        if normalize:
            features = feature_normalizer(features)

        V = cluster.k_means(features, n_clusters=self.Nk, max_iter=max_iter, n_init=redo)
        self.V.set_value(np.asarray(V[0], dtype=theano.config.floatX))


def symbolic_distance_matrix(A, B):
    """
    Defines the symbolic matrix that contains the distances between the vectors of A and B
    :param A:
    :param B:
    :return:
    """
    aa = T.sum(A * A, axis=1)
    bb = T.sum(B * B, axis=1)
    AB = T.dot(A, T.transpose(B))

    AA = T.transpose(T.tile(aa, (bb.shape[0], 1)))
    BB = T.tile(bb, (aa.shape[0], 1))

    D = AA + BB - 2 * AB
    D = T.maximum(D, 0)
    D = T.sqrt(D)
    return D


class CBoF_Input_Layer:
    def __init__(self, input, cnn, layer, level=1, pyramid=False, g=0.1, n_codewords=16):
        """
        Defines a CBoF layer for use with convolutional feature extractors
        :param input: symbolic input variable
        :param cnn: the cnn input model
        :param layer: the convolutional layer (id) to use
        :param spatial: if set to True, spatial pyramid is used
        :param g: the BoF softness variable
        :param n_codewords: number of codewords for each BoF unit
        """
        self.bof = []
        self.features = []
        self.V = []
        self.sigma = []
        self.get_features = []
        self.n = 1

        self.features_size = 0
        if level == 0 or pyramid:
            # Create the BoF object
            self.bof.append(NBoFInputLayer(g=g, feature_dimension=cnn.features_dim[layer], n_codewords=n_codewords))
            self.V.append(self.bof[0].V)
            self.sigma.append(self.bof[0].sigma)
            # Extract the representation
            self.features.append(self.bof[0].sym_histograms(cnn.get_features(layer)))
            # Compile functions for extracting feature vectors
            self.get_features.append(theano.function([input], cnn.get_features(layer)))
            # Fuse the extracted representations
            self.fused_features = self.features[0]
            # Calculate length
            self.features_size += n_codewords
        if level == 1:
            for i in range(4 ** level):
                # Create the BoF object
                self.bof.append(NBoFInputLayer(g=g, feature_dimension=cnn.features_dim[layer], n_codewords=n_codewords))
                self.V.append(self.bof[i].V)
                self.sigma.append(self.bof[i].sigma)
                # Extract the representation
                self.features.append(self.bof[i].sym_histograms(cnn.get_spatial_features(layer, i, level)))
                # Compile functions for extracting feature vectors
                self.get_features.append(theano.function([input], cnn.get_spatial_features(layer, i, level)))
            # Fuse the extracted representations
            self.fused_features = T.concatenate(tuple(self.features), axis=1)
            # Calculate length
            self.features_size += n_codewords * (4 ** level)

    def initialize(self, data, max_iter=100, redo=5, n_samples=50000, normalize=False):
        """
        Initializes each of the spatial BoF layers in the CBoF layer
        :param data: input samples
        :param max_iter: max number of iterations for the k-means algorithm
        :param redo: number to redo the clustering
        :param n_samples: number of vectors to sample for clustering
        :param normalize: use l_2 norm normalization for the feature vectors
        :return:
        """

        for i in range(len(self.bof)):
            features = []
            for x in data:
                x_in = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
                cur_features = self.get_features[i](np.float32(x_in))
                features.append(cur_features)
            features = np.asarray(features)
            features = features.reshape((features.shape[0], features.shape[2], features.shape[3]))
            self.bof[i].initialize_dictionary(features, max_iter=max_iter, redo=redo, n_samples=n_samples,
                                              normalize=normalize)
