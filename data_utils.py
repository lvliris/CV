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
        x = x.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
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


