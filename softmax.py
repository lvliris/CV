import numpy as np


def softmax_loss_naive(W, X, y, reg):
    num_train = X.shape[0]
    num_classes = W.shape[1]
    dW = np.zeros(W.shape)
    loss = 0.0

    # calculate the softmax loss by loops
    for i in range(num_train):
        score = X[i].dot(W)
        max_score = np.max(score)
        score -= max_score
        prob = np.exp(score)
        prob /= np.sum(prob)

        loss += -np.log(prob[y[i]])
        for k in range(num_classes):
            dW[:, k] += (prob[k] - (k == y[i])) * X[i]

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg*W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    num_train = X.shape[0]
    num_classes = W.shape[1]

    score = X.dot(W)
    max_score = np.max(score, axis=1, keepdims=True)
    score -= max_score
    prob = np.exp(score)
    prob /= np.sum(prob, axis=1)

    loss = np.sum(-np.log(prob[range(num_train), y])) / num_train
    loss += 0.5 * reg * np.sum(W * W)

    ind = np.zeros_like(prob)
    ind[range(num_train), y] = 1
    dW = X.T.dot(prob - ind) / num_train
    dW += reg * W

    return loss, dW
