from __future__ import print_function
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_CIFAR_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        x = dict[b'data']
        y = dict[b'labels']
        x = x.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('uint8')
        x = np.array(x)
        y = np.array(y)
        return x, y


def load_CIFAR10(dir):
    train_data, train_label, test_data, test_label = [], [], [], []
    for i in range(1, 6):
        file = os.path.join(dir, 'data_batch_%d' % (i,))
        x, y = load_CIFAR_batch(file)
        if i == 1:
            train_data = np.array(x)
            train_label = np.array(y)
        else:
            train_data = np.concatenate((train_data, x), axis=0)
            train_label = np.concatenate((train_label, y), axis=0)

    test_file = os.path.join(dir, 'test_batch')
    test_data, test_label = load_CIFAR_batch(test_file)

    return train_data, train_label, test_data, test_label


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    """
    Prepare the cifar10 data for training, sample the given number of data and subtract the mean of images
    :param num_training: the number of training examples
    :param num_validation: the number of validation examples, the total number of the first two should not exceed 50000
    :param num_test: the number of test examples, not exceed 10000
    :return: X_train, y_train, X_val, y_val, X_test, y_test, shape [N, 32, 32, 3]
    """
    X_train, y_train, X_test, y_test = load_CIFAR10('data/cifar-10-batches-py')

    # choose some data for validation
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]

    # choose some data for training
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]

    # use the formal num_test data for testing
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # reshape the data
    # X_train = X_train.reshape([X_train.shape[0], -1])
    # X_val = X_val.reshape([X_val.shape[0], -1])
    # X_test = X_test.reshape([X_test.shape[0], -1])
    # X_dev = X_dev.reshape([X_dev.shape[0], -1])

    # subtract the mean value
    '''mean_img = np.mean(X_train, axis=0).astype(np.float32)
    X_train -= mean_img
    X_val -= mean_img
    X_test -= mean_img'''

    # add a bias
    # X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    # X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    # X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    # X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == '__main__':
    file = 'data/cifar-10-batches-py/'
    X_train, y_train, X_test, y_test = load_CIFAR10(file)
    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Testing data shape: ', X_test.shape)
    print('Testing labels shape: ', y_test.shape)

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    class_num = len(classes)
    samples_pre_class = 7

    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y == y_train)
        idxs = np.random.choice(idxs, samples_pre_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * class_num + y + 1
            plt.subplot(samples_pre_class, class_num, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)

    plt.show()


