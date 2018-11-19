from linear_svm import svm_loss_vectorized
from data_utils import load_CIFAR10
import matplotlib.pyplot as plt
import numpy as np
import time


class LinearClassifier():
    def __init__(self):
        pass

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
            self.W = 0.001 * np.random.nrand(dim, num_class)

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
                print 'iteration %d / %d, loss %f' % (iter, num_iters, loss)

            return loss_history

    def predict(self, X):
        pred_y = X.dot(self.W)
        return np.max(pred_y, axis=1)

    def loss(self, X, y, reg):
        return svm_loss_vectorized(self.W, X, y, reg)

if __name__ == '__main__':
    classifier = LinearClassifier()
    X_train, y_train, X_test, y_test = load_CIFAR10('cifar-10-batches-py')

    tic = time.time()
    loss = classifier.train(X_train, y_train, learning_rate=1e-7, reg=2.5e-4, num_iters=1500, verbose=True)
    toc = time.time()
    print 'that takes %fs' % (toc - tic)

    # plot the loss
    plt.plot(loss)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.show()

    y_train_pred = classifier.predict(X_train)
    print 'training accuracy: %f' % (np.mean(y_train == y_train_pred), )
    y_test_pred = classifier.predict(X_test)
    print 'testing accuracy: %f' % (np.mean(y_test == y_test_pred), )
