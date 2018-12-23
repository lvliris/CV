import numpy as np
import matplotlib.pyplot as plt
from gradient_check import gradient_check_sparse
from softmax import softmax_loss_naive
from linear_svm import svm_loss_naive


class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, num_classes, std=1e-1):
        self.momentum = 0.95
        self.learning_rate = 1e-3

        self.param = {}
        self.param['W1'] = 0.001 * np.random.randn(input_size, hidden_size)
        self.param['b1'] = 0.001 * np.random.randn(hidden_size)
        self.param['W2'] = 0.001 * np.random.randn(hidden_size, num_classes)
        self.param['b2'] = 0.001 * np.random.randn(num_classes)

        self.param['dW1'] = np.zeros([input_size, hidden_size])
        self.param['db1'] = np.zeros(hidden_size)
        self.param['dW2'] = np.zeros([hidden_size, num_classes])
        self.param['db2'] = np.random.randn(num_classes)

        self.param['mW1'] = np.zeros([input_size, hidden_size])
        self.param['mb1'] = np.zeros(hidden_size)
        self.param['mW2'] = np.zeros([hidden_size, num_classes])
        self.param['mb2'] = np.random.randn(num_classes)

    def loss(self, X, y, reg=0.05):
        N = X.shape[0]
        # calculate the media results use ReLU activation function
        z1 = X.dot(self.param['W1']) + self.param['b1']
        a1 = np.maximum(z1, 0)

        # calculate the output score
        scores = a1.dot(self.param['W2']) + self.param['b2']
        max_scores = np.max(scores, axis=1, keepdims=True)

        # calculate the loss and regularization
        exp_score = np.exp(scores - max_scores)
        probs = exp_score / np.sum(exp_score, axis=1, keepdims=True)
        loss = np.sum(-np.log(probs[range(N), y] + 1e-8)) / N

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

        # momentum
        self.param['mW2'] = self.momentum * self.param['mW2'] + self.param['dW2']
        self.param['mb2'] = self.momentum * self.param['mb2'] + self.param['db2']

        self.param['mW1'] = self.momentum * self.param['mW1'] + self.param['dW1']
        self.param['mb1'] = self.momentum * self.param['mb1'] + self.param['db1']

        return loss

    def train(self, X_train, y_train, X_val, y_val, learning_rate=1e-5, learning_rate_decay=0.95,
              reg=0.001, num_iters=1000, batch_size=200, verbose=False):
        num_train = X_train.shape[0]
        self.momentum = learning_rate_decay
        self.learning_rate = learning_rate

        loss_history = []
        train_accuracy_history = []
        val_accuracy_history = []
        for i in range(num_iters):
            # sample the batch data
            batch_indices = np.random.choice(np.arange(num_train), batch_size)
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            loss = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # update the parameters
            self.param['W1'] -= self.learning_rate * self.param['mW1']
            self.param['b1'] -= self.learning_rate * self.param['mb1']
            self.param['W2'] -= self.learning_rate * self.param['mW2']
            self.param['b2'] -= self.learning_rate * self.param['mb2']

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
        max_scores = np.max(scores, axis=1, keepdims=True)

        # calculate the probabilities
        exp_score = np.exp(scores - max_scores)
        probs = exp_score / np.sum(exp_score, axis=1, keepdims=True)
        max_probs = np.max(probs, axis=1, keepdims=True)

        x, y = np.where(probs == max_probs)

        return y


class Layer(object):
    def __init__(self):
        self.W, self.b = None, None

    def forward(self, bottom):
        pass

    def backward(self, top):
        pass


class FullConnected(Layer):
    def __init__(self, input_size, output_size):
        super(FullConnected, self).__init__()
        self.bottom = None
        self.type = 'fc'
        self.momentum = 0.9
        self.W = 0.01 * np.random.randn(input_size, output_size)
        self.b = 0.01 * np.random.randn(output_size)
        self.dW = 0.01 * np.random.randn(input_size, output_size)
        self.db = 0.01 * np.random.randn(output_size)
        self.mW = np.zeros([input_size, output_size])
        self.mb = np.zeros(output_size)

    def forward(self, bottom):
        self.bottom = bottom.reshape([bottom.shape[0], -1])
        assert self.bottom.shape[1] == self.W.shape[0]
        # top = top.reshape([bottom.shape[0], self.W.shape[1]])
        top = np.dot(self.bottom, self.W) + self.b

        return top

    def backward(self, top):
        assert top.shape[1] == self.W.shape[1]
        # bottom = bottom.reshape([top.shape[0], self.W.shape[0]])
        bottom = np.dot(top, self.W.T)

        self.dW = np.dot(self.bottom.T, top)
        self.db = np.sum(top, axis=0)

        self.mW = self.momentum * self.mW + self.dW
        self.mb = self.momentum * self.mb + self.db

        return bottom


class Dropout(Layer):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p
        self.dropout_indice = None

    def forward(self, bottom, is_training=True):
        if is_training:
            num_data = bottom.shape[1]
            self.dropout_indice = np.random.choice(num_data, int(num_data * self.p), replace=False)
            top = bottom
            top[:, self.dropout_indice] = 0
        else:
            top = self.p * bottom

        return top

    def backward(self, top):
        bottom = top
        bottom[:, self.dropout_indice] = 0

        return bottom


class BatchNormalization(Layer):
    def __init__(self, shape):
        super(BatchNormalization, self).__init__()
        self.alpha = 0.9
        self.W = np.zeros(shape)
        self.b = np.zeros(shape)
        self.dW = np.zeros(shape)
        self.db = np.zeros(shape)
        self.mean = np.zeros(shape)
        self.var = np.zeros(shape)
        self.batch_mean = np.zeros(shape)
        self.batch_var = np.zeros(shape)
        self.bottom = None

    def forward(self, bottom, is_training=True):
        self.bottom = bottom
        if is_training:
            self.batch_mean = np.mean(bottom, axis=0)
            self.batch_var = np.mean((bottom - self.batch_mean)*(bottom - self.batch_mean))
            # calculate the expectation of data by moving average
            self.mean = self.alpha * self.mean + self.batch_mean
            self.var = self.alpha * self.var + self.batch_var
        else:
            self.batch_mean = self.mean
            self.batch_var = self.var

        self.x_norm = (bottom - self.batch_mean) / np.sqrt(self.batch_var + 1e-8)
        top = self.W * self.x_norm + self.b

        return top

    def backward(self, top):
        dx_norm = top * self.W
        dvar = np.sum(-dx_norm * (self.bottom - self.batch_mean) * 0.5 * np.power(self.batch_var + 1e-8, -1.5), axis=0)
        dmean = np.sum(-dx_norm / np.sqrt(self.batch_var + 1e-8)) + dvar * np.mean(-2 * (self.bottom - self.batch_mean), axis=0)
        bottom = dx_norm / np.sqrt(self.batch_var + 1e-8) + dvar * np.mean(2*(self.bottom - self.batch_mean), axis=0) + np.mean(dmean, axis=0)

        self.dW = np.sum(top * self.x_norm, axis=0)
        self.db = np.sum(top, axis=0)

        return bottom


class ReLU(Layer):
    def __init__(self):
        super(ReLU, self).__init__()
        self.type = 'relu'
        self.indice = None

    def forward(self, bottom):
        self.indice = bottom < 0
        # top = top.reshape(bottom.shape)
        top = np.maximum(bottom, 0)

        return top

    def backward(self, top):
        # bottom = bottom.reshape(top.shape)
        bottom = top
        bottom[self.indice] = 0

        return bottom


class Softmax(Layer):
    def __init__(self):
        super(Softmax, self).__init__()
        self.type = 'softmax'

    def forward(self, bottom):
        max_score = np.max(bottom, axis=1, keepdims=True)
        self.prob = np.exp(bottom - max_score)
        self.prob /= np.sum(self.prob, axis=1, keepdims=True)
        # top = top.reshape(bottom.shape)
        top = self.prob

        return top

    def backward(self, top):
        ak = self.prob * top
        akv = np.sum(ak, axis=1, keepdims=True)
        bottom = ak - self.prob * akv

        # dx = np.dot(self.prob.T, self.prob)
        # diag = np.diag(self.prob[0])
        # dx = diag - dx
        # bottom = bottom.reshape(top.shape)
        # bottom = np.dot(top, dx)

        return bottom


class SoftmaxWithLoss(Layer):
    def __init__(self):
        super(SoftmaxWithLoss, self).__init__()
        self.type = 'softmax-with-loss'
        self.data, self.label = None, None

    def forward(self, bottom):
        self.data = bottom[0]
        self.label = bottom[1].astype(int)
        max_score = np.max(self.data, axis=1, keepdims=True)
        self.prob = np.exp(self.data - max_score)
        self.prob /= np.sum(self.prob, axis=1, keepdims=True)
        N = self.data.shape[0]
        top = -np.sum(np.log(self.prob + 1e-8) * self.label) / N

        return top

    def backward(self, top):
        bottom = self.prob
        bottom -= self.label
        bottom = top * bottom / self.data.shape[0]

        return bottom


class CrossEntropyLoss(Layer):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.type = 'cross-entropy'
        self.data, self.label = None, None

    def forward(self, bottom):
        self.data = bottom[0]
        self.label = bottom[1]
        # top = top.reshape(1)
        top = -np.sum(self.label * np.log(self.data + 1e-5)) / self.data.shape[0]

        return top

    def backward(self, top):
        # bottom.reshape(self.data.shape)
        bottom = -top * self.label / (self.data + 1e-5) / self.data.shape[0]

        return bottom


class Net(object):
    def forward(self, data, label):
        pass

    def backward(self, loss):
        pass

    def update(self):
        pass

    def train(self, data, label, val_data, val_label, learning_rate=1e-3, reg=0.5, iters=1000, verbose=True):
        pass

    def predict(self, data):
        pass


class FullConnectedNet(Net):
    def __init__(self, input_size, layer_size, output_size):
        self.learning_rate = 1e-3
        self.input_size = input_size
        self.output_size = output_size
        self.W2 = 0.0
        self.loss = 0.0
        self.layers = []
        last_size = input_size
        for hidden_size in layer_size:
            layer = FullConnected(last_size, hidden_size)
            self.layers.append(layer)
            layer = ReLU()
            self.layers.append(layer)
            last_size = hidden_size

        # dropout layer
        layer = Dropout()
        self.layers.append(layer)

        # output layer
        layer = FullConnected(last_size, output_size)
        self.layers.append(layer)
        layer = Softmax()
        self.layers.append(layer)

        # loss layer
        self.loss_layer = CrossEntropyLoss()
        # self.loss_layer = SoftmaxWithLoss()

    def forward(self, data, label):
        bottom = data
        self.W2 = 0.0
        for layer in self.layers:
            top = layer.forward(bottom)
            bottom = top
            if layer.W is not None:
                self.W2 += np.sum(layer.W * layer.W)

        # loss
        # print 'data.shape, label.shape:', bottom.shape, label.shape
        bottom = np.stack((bottom, label), axis=0)
        self.loss = self.loss_layer.forward(bottom)

        return self.loss

    def backward(self, loss):
        # self.loss_layer.backward(loss, bottom)
        bottom = self.loss_layer.backward(loss)
        top = bottom

        for layer in reversed(self.layers):
            bottom = layer.backward(top)
            top = bottom

    def update(self):
        for layer in self.layers:
            if layer.W is not None:
                layer.W -= self.learning_rate * layer.dW
                layer.b -= self.learning_rate * layer.db

    def train(self, data, label, val_data, val_label, learning_rate=1e-3, learning_rate_decay=0.95,
              reg=0.5, batch_size=200, num_iters=1000, verbose=True):
        assert data.shape[1] == self.input_size
        self.learning_rate = learning_rate
        num_train = data.shape[0]
        state = {}
        loss_history = []
        train_accuracy_history = []
        val_accuracy_history = []

        num_data = data.shape[0]
        label_mat = np.zeros([num_data, self.output_size])
        label_mat[range(num_data), label] = 1
        for i in range(num_iters):
            batch_indice = np.random.choice(np.arange(num_train), batch_size)
            data_batch = data[batch_indice]
            label_batch = label[batch_indice]
            self.forward(data, label_mat)
            self.loss += 0.5 * reg * self.W2
            loss_history.append(self.loss)
            self.backward(1)
            self.update()

            if verbose and i % 100 == 0:
                print('iteration %d, loss: %f' % (i, self.loss))
                y_train_pred = self.predict(data)
                train_accuracy_history.append(np.mean(label == y_train_pred))
                y_val_pred = self.predict(val_data)
                val_accuracy_history.append(np.mean(val_label == y_val_pred))

            # if i == 2000:
            #    self.learning_rate *= 0.1

            state = {}
            state['loss_history'] = loss_history
            state['train_accuracy_history'] = train_accuracy_history
            state['val_accuracy_history'] = val_accuracy_history

        return state

    def predict(self, data):
        bottom = data
        top = np.array([])
        for layer in self.layers:
            top = layer.forward(bottom)
            bottom = top

        max_prob = np.max(top, axis=1, keepdims=True)
        x, y = np.where(top == max_prob)

        return y


def init_toy_model(input_size, hidden_size, output_size):
    np.random.seed(0)
    return TwoLayerNet(input_size, hidden_size, output_size, std=1e-1)


def init_toy_data(num_input, input_size):
    np.random.seed(1)
    X = 10 * np.random.randn(num_input, input_size)
    y = np.array([0, 1, 2, 2, 1])
    # y = np.array([0])
    return X, y


if __name__ == '__main__':
    input_size = 4
    hidden_size = 10
    num_classes = 3
    num_input = 5

    a = np.array([1])
    b = np.array([2])
    print a, b

    # net = init_toy_model(input_size, hidden_size, num_classes)
    net = FullConnectedNet(input_size, [100], num_classes)
    X, y = init_toy_data(num_input, input_size)

    # W = 0.0001 * np.random.randn(input_size, num_classes)
    # W = np.zeros([input_size, num_classes])
    # loss, grad = svm_loss_naive(W, X, y, 0.0)
    # print loss
    # f = lambda w: svm_loss_naive(w, X, y, 0.0)[0]
    # gradient_check_sparse(f, W, grad, 10)

    state = net.train(X, y, X, y, learning_rate=1e-3, reg=0, num_iters=20000, verbose=True)
    loss = state['loss_history']
    train_acc = state['train_accuracy_history']
    val_acc = state['val_accuracy_history']
    pred_y = net.predict(X)
    print(X)
    print(y)
    print(pred_y)
    plt.subplot(211)
    plt.plot(loss)
    plt.subplot(212)
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.show()
