import numpy as np
import matplotlib.pyplot as plt
from gradient_check import gradient_check_sparse
from softmax import softmax_loss_naive
from linear_svm import svm_loss_naive


class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, num_classes, std=1e-1):
        self.param = {}
        self.param['W1'] = 0.001 * np.random.randn(input_size, hidden_size)
        self.param['b1'] = 0.001 * np.random.randn(hidden_size)
        self.param['W2'] = 0.001 * np.random.randn(hidden_size, num_classes)
        self.param['b2'] = 0.001 * np.random.randn(num_classes)

        self.param['dW1'] = np.zeros([input_size, hidden_size])
        self.param['db1'] = np.zeros(hidden_size)
        self.param['dW2'] = np.zeros([hidden_size, num_classes])
        self.param['db2'] = np.random.randn(num_classes)

    def loss(self, X, y, reg=0.05):
        N = X.shape[0]
        # calculate the media results use ReLU activation function
        z1 = X.dot(self.param['W1']) + self.param['b1']
        a1 = np.maximum(z1, 0)

        # calculate the output score
        scores = a1.dot(self.param['W2']) + self.param['b2']

        # calculate the loss and regularization
        exp_score = np.exp(scores)
        probs = exp_score / np.sum(exp_score, axis=1, keepdims=True)
        loss = np.sum(-np.log(probs[range(N), y])) / N

        # calculate the gradient
        dscores = probs
        dscores[range(N), y] -= 1
        dscores /= N

        self.param['dW2'] = np.dot(a1.T, dscores)
        self.param['db2'] = np.sum(dscores, axis=0)
        da1 = np.dot(dscores, self.param['W2'].T)
        # activation function gradient
        da1[a1 <= 0] = 0

        self.param['dW1'] = np.dot(X.T, da1)
        self.param['db1'] = np.sum(da1, axis=0)

        # regularization
        self.param['dW1'] += reg * self.param['W1']
        self.param['dW2'] += reg * self.param['W2']

        return loss

    def train(self, X_train, y_train, X_val, y_val, learning_rate=1e-5, reg=0.001, num_iters=1000, batch_size=200, verbose=False):
        num_train = X_train.shape[0]
        batch_indices = np.random.choice(np.arange(num_train), batch_size)
        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]

        loss_history = []
        train_accuracy_history = []
        val_accuracy_history = []
        for i in range(num_iters):
            loss = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # update the parameters
            self.param['W1'] -= learning_rate * self.param['dW1']
            self.param['b1'] -= learning_rate * self.param['db1']
            self.param['W2'] -= learning_rate * self.param['dW2']
            self.param['b2'] -= learning_rate * self.param['db2']

            if verbose and i % 100 == 0:
                print('iteration %d, loss: %f' % (i, loss))
                y_train_pred = self.predict(X_train)
                train_accuracy_history.append(np.mean(y_train == y_train_pred))
                y_val_pred = self.predict(X_val)
                val_accuracy_history.append(np.mean(y_val == y_val_pred))

        state = {}
        state['loss_history'] = loss_history
        state['train_accuracy_history'] = train_accuracy_history
        state['val_accuracy_history'] = val_accuracy_history

        return state

    def predict(self, X):
        # calculate the media results use ReLU activation function
        z1 = X.dot(self.param['W1']) + self.param['b1']
        a1 = np.maximum(z1, 0)

        # calculate the output score
        scores = a1.dot(self.param['W2']) + self.param['b2']

        # calculate the probabilities
        exp_score = np.exp(scores)
        probs = exp_score / np.sum(exp_score, axis=1, keepdims=True)
        max_probs = np.max(probs, axis=1, keepdims=True)

        x, y = np.where(probs == max_probs)

        return y


def init_toy_model(input_size, hidden_size, output_size):
    np.random.seed(0)
    return TwoLayerNet(input_size, hidden_size, output_size, std=1e-1)


def init_toy_data(num_input, input_size):
    np.random.seed(1)
    X = 10 * np.random.randn(num_input, input_size)
    y = np.array([0, 1, 2, 2, 1])
    return X, y


if __name__ == '__main__':
    input_size = 4
    hidden_size = 10
    num_classes = 3
    num_input = 5

    net = init_toy_model(input_size, hidden_size, num_classes)
    X, y = init_toy_data(num_input, input_size)

    # W = 0.0001 * np.random.randn(input_size, num_classes)
    # W = np.zeros([input_size, num_classes])
    # loss, grad = svm_loss_naive(W, X, y, 0.0)
    # print loss
    # f = lambda w: svm_loss_naive(w, X, y, 0.0)[0]
    # gradient_check_sparse(f, W, grad, 10)

    loss = net.train(X, y, learning_rate=1e-1, reg=5e-6, iters=100, verbose=False)
    plt.plot(loss)
    plt.show()

