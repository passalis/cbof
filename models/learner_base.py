import numpy as np
from time import time
from sklearn.metrics import accuracy_score
import time
import cv2


class LearnerBase():
    def __init__(self):
        self.train_fn = None
        self.test_fn = None
        self.lr = None

        self.best_param_values = []
        self.params = []

    def save_validation_parameters(self):
        """
        Saves the best parameters found during the validation
        """
        self.best_param_values = []
        for i, param in enumerate(self.params):
            self.best_param_values.append(param.get_value())

    def restore_validation_parameters(self):
        """
        Restores the best parameters
        """
        if len(self.best_param_values) > 0:
            for i, param in enumerate(self.params):
                param.set_value(self.best_param_values[i])

    def train_model(self, data, labels, batch_size=32, n_iters=10, validation_data=None, validation_labels=None):

        loss = []
        idx = np.arange(data.shape[0])
        best_val_acc = 0

        for i in range(n_iters):
            np.random.shuffle(idx)
            cur_loss = 0
            n_batches = data.shape[0] / batch_size
            start_time = time.time()

            # Iterate mini-batches
            for j in range(n_batches):
                cur_idx = np.sort(idx[j * batch_size:(j + 1) * batch_size])
                cur_data = data[cur_idx]
                cur_labels = labels[cur_idx]
                cur_loss += self.train_fn(cur_data, cur_labels) * cur_data.shape[0]

            # Last batch
            if n_batches * batch_size < data.shape[0]:
                # for cur_scale in scales:
                cur_idx = np.sort(idx[n_batches * batch_size:])
                cur_data = data[cur_idx]
                cur_labels = labels[cur_idx]
                cur_loss += self.train_fn(cur_data, cur_labels) * cur_data.shape[0]

            loss.append(cur_loss / float(data.shape[0]))
            elapsed_time = time.time() - start_time

            print "Epoch %d loss = %5.4f, cur_time: %6.1f s time_left: %8.1f s" % \
                  (i + 1, loss[-1], elapsed_time, (n_iters - i) * elapsed_time)

            if validation_data is not None:
                val_acc = self.test_model(validation_data, validation_labels)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    print "New best found!", val_acc
                    self.save_validation_parameters()

        if validation_data is not None:
            self.restore_validation_parameters()

        return loss

    def test_model(self, data, labels, scale=1, batch_size=100):
        """

        :param data: images for testing
        :param labels: classes of the images
        :param scale: the scale used for the testing (only if global pooling/cbof is used)
        :param batch_size: batch size to be used for the testing
        :return:
        """
        predicted_labels = []

        # Resize images if needed
        if scale != 1:
            img_size = [int(x * scale) for x in data.shape[2:]]
            new_data = np.zeros((data.shape[0], data.shape[1], img_size[0], img_size[1]))
            for k in range(data.shape[0]):
                new_data[k] = resize_image(data[k], img_size)
            data = np.float32(new_data)

        n_batches = data.shape[0] / batch_size

        # Iterate mini-batches
        for j in range(n_batches):
            cur_data = data[j * batch_size:(j + 1) * batch_size]
            predicted_labels.extend(self.test_fn(cur_data))
        # Last batch
        if n_batches * batch_size < data.shape[0]:
            cur_data = data[n_batches * batch_size:]
            predicted_labels.extend(self.test_fn(cur_data))
        predicted_labels = np.asarray(predicted_labels)

        acc = accuracy_score(labels, predicted_labels)
        return acc


def resize_image(img, size):
    img = img.transpose((1, 2, 0))
    img = cv2.resize(img, (size[0], size[1]))

    if len(img.shape) == 2:
        img = img.reshape((1, img.shape[0], img.shape[1]))
    else:
        img = img.transpose((2, 0, 1))
    return img
