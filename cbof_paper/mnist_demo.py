import numpy as np
import sklearn.utils

from model.cbof import CBoF
from model.cnn import CNN_Simple
from model.datasets import load_mnist, resize_mnist_data


# Set the path to mnist.pkl.gz before running the code
# Download mnist.pkl.gz from http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

def run_demo_mnist(model='cbof', n_iters=50, seed=1, ):

    # Load mnist data
    train_data, valid_data, test_data, train_labels, valid_labels, test_labels = load_mnist(
        dataset='/home/nick/Data/Datasets/mnist.pkl.gz')

    if model != 'plain':
        train_data_20 = resize_mnist_data(train_data, 20, 20)
        train_data_24 = resize_mnist_data(train_data, 24, 24)
        train_data_32 = resize_mnist_data(train_data, 32, 32)
        train_data_36 = resize_mnist_data(train_data, 36, 36)
        test_data_20 = resize_mnist_data(test_data, 20, 20)

    # Set seeds for reproducibility
    sklearn.utils.check_random_state(seed)
    np.random.seed(seed)

    eta = 0.0001
    if model == 'cbof':
        cnn = CBoF(learning_rate=eta, n_classes=10, bof_layer=(1, True, 64), hidden_neurons=(1000,))
        cnn.init_bof(train_data[:50000, :])
    elif model == 'spp':
        cnn = CNN_Simple(learning_rate=eta, hidden_neurons=(1000,), n_classes=10, use_spatial_pooling=True,
                         pool_dims=[1, 2])
    elif model == 'gmp':
        cnn = CNN_Simple(learning_rate=eta, hidden_neurons=(1000,), n_classes=10, use_spatial_pooling=True,
                         pool_dims=[1])
    elif model == 'plain':
        cnn = CNN_Simple(learning_rate=eta, hidden_neurons=(1000,), n_classes=10, use_spatial_pooling=False)

    best_valid, test_acc, best_iter = 0, 0, 0

    for i in range(n_iters):

        if model != 'plain':
            cnn.train_model(train_data_20, train_labels, batch_size=64)
            cnn.train_model(train_data_24, train_labels, batch_size=64)
            cnn.train_model(train_data_32, train_labels, batch_size=64)
            cnn.train_model(train_data_36, train_labels, batch_size=64)
        loss = cnn.train_model(train_data, train_labels, batch_size=64)
        print("Iter: ", i, ", loss: ", loss)

        # Get validation accuracy
        valid_acc = cnn.test_model(valid_data, valid_labels)
        if valid_acc > best_valid:
            best_valid = valid_acc
            best_iter = i
            # Test the model!
            test_acc = cnn.test_model(test_data, test_labels)
            test_acc_20 = 0
            if model != 'plain':
                test_acc_20 = cnn.test_model(test_data_20, test_labels)
            print("New validation best found, valid acc = ", valid_acc, " iter = ", i)
            print(test_acc, test_acc_20)

    print("Evaluated model = ", model)
    print("Best err = ", 100 - test_acc, "% found @ iter = ", best_iter)
    if model != 'plain':
        print("Err (20x20): ", 100 - test_acc_20)


run_demo_mnist(model='plain')
run_demo_mnist(model='gmp')
run_demo_mnist(model='spp')
run_demo_mnist(model='cbof')

