import numpy as np
from tqdm import tqdm


class Base_Learner:
    def __init__(self):
        self.train_fn = None
        self.train_mlp_fn = None
        self.test_fn = None

    def train_model(self, train_data, train_labels, batch_size=32, pre_train=False):
        """
        Trains the model
        :param train_data:
        :param train_labels:
        :param batch_size:
        :param pre_train:
        :return:
        """
        n_batches = int(np.floor(train_data.shape[0] / batch_size))
        loss = 0
        for i in tqdm(range(n_batches)):
            cur_data = train_data[i * batch_size:(i + 1) * batch_size, :]
            cur_labels = train_labels[i * batch_size:(i + 1) * batch_size]
            if pre_train:
                cur_loss = self.train_mlp_fn(np.float32(cur_data), np.int32(cur_labels))
            else:
                cur_loss = self.train_fn(np.float32(cur_data), np.int32(cur_labels))
            loss += cur_loss * batch_size

        if n_batches * batch_size < train_data.shape[0]:
            cur_data = train_data[n_batches * batch_size:, :]
            cur_labels = train_labels[n_batches * batch_size:]
            if pre_train:
                cur_loss = self.train_mlp_fn(np.float32(cur_data), np.int32(cur_labels))
            else:
                cur_loss = self.train_fn(np.float32(cur_data), np.int32(cur_labels))
            loss += cur_loss * train_data.shape[0]
        loss = loss / float(train_data.shape[0])
        return loss

    def test_model(self, test_data, test_labels, batch_size=32):
        """
        Predicts the labels and returns the accuracy and the precision
        :param test_data:
        :param test_labels:
        :param batch_size:
        :return:
        """
        labels = np.zeros((0,))
        n_batches = int(np.floor(test_data.shape[0] / batch_size))

        for i in range(n_batches):
            cur_data = test_data[i * batch_size:(i + 1) * batch_size, :]
            labels = np.hstack((labels, self.test_fn(np.float32(cur_data))))

        if n_batches * batch_size < test_data.shape[0]:
            cur_data = test_data[n_batches * batch_size:, :]
            labels = np.hstack((labels, self.test_fn(np.float32(cur_data))))

        return 100 * np.mean(test_labels == labels)
