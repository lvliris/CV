from linear_svm import svm_loss_vectorized
from softmax import softmax_loss_vectorized
from neural_net import TwoLayerNet, FullConnectedNet, ConvolutionalNet
from data_utils import load_CIFAR10
import matplotlib.pyplot as plt
import numpy as np
import time
import math


class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
        """
        Train this linear classifier with stochastic gradient descent.
        :param X: numpy array, shape (N, D), training examples
        :param y: numpy array, shape (N,), labels
        :param learning_rate: float
        :param reg: float
        :param num_iters: integer
        :param batch_size: integer
        :param verbose: boolean, if true print the optimize process
        :return: list, store the loss of every iteration
        """
        num_train, dim = X.shape
        num_class = np.max(y) + 1

        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_class)

        # use SGD to optimize W
        loss_history = []
        for iter in range(num_iters):
            # choose a mini_batch with replacement, it's faster
            batch_indice = np.random.choice(num_train, batch_size)
            X_batch = X[batch_indice, :]
            y_batch = y[batch_indice]

            # perform the loss calculation and update the weights
            loss, dw = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)
            self.W -= learning_rate * dw

            # print the optimize process
            if verbose and iter % 100 == 0:
                print('iteration %d / %d, loss %f' % (iter, num_iters, loss))

        return loss_history

    def predict(self, X):
        pred_y = X.dot(self.W)
        max_y = np.max(pred_y, axis=1)
        max_y = max_y.reshape(max_y.shape[0], -1)
        x, y = np.where(pred_y == max_y)
        return y

    def loss(self, X, y, reg):
        return svm_loss_vectorized(self.W, X, y, reg)


class LinearSVM(LinearClassifier):
    def loss(self, X, y, reg):
        return svm_loss_vectorized(self.W, X, y, reg)


class Softmax(LinearClassifier):
    def loss(self, X, y, reg):
        return softmax_loss_vectorized(X, y, reg)


if __name__ == '__main__':
    # classifier = LinearClassifier()
    X_train, y_train, X_test, y_test = load_CIFAR10('data/cifar-10-batches-py')

    num_training = 9900
    num_validation = 100
    num_test = 1000
    num_dev = 50

    # choose some data for training
    '''mask = np.random.choice(X_train.shape[0], 5000, replace=False)
    X_train = X_train[mask]
    y_train = y_train[mask]'''

    # choose some data for validation
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]

    # choose some data for development randomly
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]

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
    mean_img = np.mean(X_train, axis=0).astype(np.float32)
    X_train -= mean_img
    X_val -= mean_img
    X_test -= mean_img
    X_dev -= mean_img

    # add a bias
    # X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    # X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    # X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    # X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

    # classifier = TwoLayerNet(X_train.shape[1], 500, 10)
    # classifier = FullConnectedNet(X_train.shape[1], [100, 100, 100], 10)
    classifier = ConvolutionalNet(X_train.shape[1:], [(10, 3, 3)], 10)
    tic = time.time()
    # overfit one data point
    # X = X_train[0:2]
    # y = y_train[0:2]
    # state = classifier.train(X, y, X, y, batch_size=2, num_iters=1000, verbose=True)
    state = classifier.train(X_train, y_train, X_val, y_val,
                             learning_rate=1e-2, learning_rate_decay=0.95,
                             reg=0, num_iters=2000, batch_size=500,
                             verbose=True)
    toc = time.time()
    print('that takes %fs' % (toc - tic))

    # perform prediction
    y_train_pred = classifier.predict(X_train)
    print('training accuracy: %f' % (np.mean(y_train == y_train_pred), ))
    y_test_pred = classifier.predict(X_test)
    print('testing accuracy: %f' % (np.mean(y_test == y_test_pred), ))

    # plot the loss and accuracy
    plt.subplot(2, 1, 1)
    plt.plot(state['loss_history'])
    plt.title('loss history')
    plt.xlabel('iterations')
    plt.ylabel('loss')

    plt.subplot(2, 1, 2)
    plt.plot(state['train_accuracy_history'], label='train')
    plt.plot(state['val_accuracy_history'], label='val')
    plt.title('accuracy history')
    plt.xlabel('epochs')
    plt.ylabel('classification accuracy')
    plt.show()

    # adjust the hyperparameters using validation data set
    learning_rates = [2e-3, 1e-3, 1e-4]
    regularization_strengths = [3e-1, 5e-1, 3, 5, 10]
    
    results = {}
    best_val = -1
    best_svm = None

    # X_val = X_train[-10000::, :]
    # y_val = y_train[-10000::]
    # X_train = X_train[:-10000:, :]
    # y_train = y_train[:-10000:]

    for lr in learning_rates:
        for reg in regularization_strengths:
            svm = TwoLayerNet(X_train.shape[1], 500, 10)
            svm.train(X_train, y_train, X_val, y_val,
                      learning_rate=lr, reg=reg,
                      num_iters=1000, batch_size=2000,
                      verbose=True)
            y_train_pred = svm.predict(X_train)
            accuracy_train = np.mean(y_train == y_train_pred)
            y_val_pred = svm.predict(X_val)
            accuracy_val = np.mean(y_val == y_val_pred)

            results[(lr, reg)] = (accuracy_train, accuracy_val)

            if best_val < accuracy_val:
                best_val = accuracy_val
                best_svm = svm

    for lr, reg in sorted(results):
        accuracy_train, accuracy_val = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (lr, reg, accuracy_train, accuracy_val))

    print('best validation accuracy achieved during cross-validation: %f' % best_val)

    # visualize the result
    x_scatter = [math.log10(x[0]) for x in results]
    y_scatter = [math.log10(x[1]) for x in results]

    marker_size = 100

    # plot training accuracy
    gray = [results[x][0] for x in results]
    plt.subplot(2, 1, 1)
    plt.scatter(x_scatter, y_scatter, marker_size, c=gray)
    plt.colorbar()
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strengths')
    plt.title('cifar-10 training accuracy')

    # plot validation accuracy
    gray = [results[x][1] for x in results]
    plt.subplot(2, 1, 2)
    plt.scatter(x_scatter, y_scatter, marker_size, c=gray)
    plt.colorbar()
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strengths')
    plt.title('cifar-10 validation accuracy')

    plt.tight_layout()
    plt.show()

    # evaluate the performance of the best SVM classifier on test set
    y_test_pred = best_svm.predict(X_test)
    test_accuracy = np.mean(y_test_pred == y_test)
    print('linear SVM on raw pixels final test set accuracy: %f' % test_accuracy)

    # visualize the weights
    w = best_svm.param['W1']   # strip out the bias
    w = w.reshape(32, 32, 3, 10)
    w_min, w_max = np.min(w), np.max(w)
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    for i in range(len(classes)):
        plt.subplot(5, 2, i+1)

        # rescale the data into 0~255
        wimg = (w[:, :, :, i].squeeze() - w_min) * 255.0 / (w_max - w_min)
        plt.imshow(wimg.astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])
    plt.show()
