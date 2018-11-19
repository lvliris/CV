import numpy as np


def svm_loss_naive(W, X, y, reg):
    """
    realize SVM loss by loop
    input dimension D, classes C, training examples N
    input:
    -W: numpy array, shape (D, C), weights
    -X: numpy array, shape (N, D), data mini-batch
    -y: numpy array, shape (N,), labels. y[i] = c means the label of x[i] is c
    -reg: float, normalization factor

    output a tuple:
    - loss, float
    - the gradient of W, same shape as W
    """
    dW = np.zeros(W.shape)
    num_class = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_class):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i].T
                dW[:, y[i]] += -X[i].T

    loss /= num_train
    dW /= num_train

    # normalization
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """realize loss by vector, the input and output are the same as svm_loss_naive"""
    dW = np.zeros(W.shape)
    num_class = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    score = X.dot(W)
    correct_class_score = score[np.arange(num_train), y]
    correct_class_score = np.reshape(correct_class_score, (num_train, -1))
    margin = scores - correct_class_score + 1
    margin = np.maximum(0, margin)
    margin[np.arange(num_train), y] = 0
    loss = np.sum(margin) / num_train
    loss += 0.5 * reg * np.sum(W * W)

    margin[margin > 0] = 1
    margin[np.arange(num_train), y] = -np.sum(margin, axis=1)
    dW += np.dot(X.T, margin)/num_train + reg * W

    return loss, dW
