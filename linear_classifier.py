from linear_svm import svm_loss_vectorized
from data_utils import load_CIFAR10
import matplotlib.pyplot as plt
import numpy as np
import time
import math


class LinearClassifier():
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

    # perform prediction
    y_train_pred = classifier.predict(X_train)
    print 'training accuracy: %f' % (np.mean(y_train == y_train_pred), )
    y_test_pred = classifier.predict(X_test)
    print 'testing accuracy: %f' % (np.mean(y_test == y_test_pred), )

    # adjust the hypeprameters using validation data set
    learning_rates = [2e-7, 0.75e-7, 1.5e-7, 1.25e-7] 
    regularization_strengths = [3e-4, 3.25e-4, 3.5e-4, 4e-4, 4.25e-4]
    
    results = {}
    best_val = -1
    best_svm = None

    X_val = X_train[-10000, ::]
    y_val = y_train[-10000, ::]
    X_train = X_train[:-10000, :]
    y_train = y_train[:-10000, :]

    for lr in learning_rates:
        for reg in regularization_strengths:
            svm = LinearClassifier()
            svm.train(X_train, y_train, learning_rate=lr, reg=reg, num_iters=100, verbose=True)
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
    w = best_svm.W[:-1, :]   # strip out the bias
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