import numpy as np


def load_mnist(dataset='/home/nick/Data/Datasets/mnist.pkl.gz'):
    """
    Loads the mnist dataset
    :return:
    """
    import gzip
    import pickle

    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    train_data = train_set[0].reshape((-1, 1, 28, 28))
    valid_data = valid_set[0].reshape((-1, 1, 28, 28))
    test_data = test_set[0].reshape((-1, 1, 28, 28))
    return train_data, valid_data, test_data, train_set[1], valid_set[1], test_set[1]


def resize_mnist_data(images, new_size_a, new_size_b=None):
    """
    Resizes a set of images
    :param images:
    :param new_size:
    :return:
    """
    from skimage.transform import resize

    if new_size_b is None:
        new_size_b = new_size_a

    resized_data = np.zeros((images.shape[0], 1, new_size_a, new_size_b))
    for i in range(len(images)):
        resized_data[i, 0, :, :] = resize(images[i, 0, :, :], (new_size_a, new_size_b))
    return np.float32(resized_data)




