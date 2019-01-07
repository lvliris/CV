import numpy as np


def relative_error(x1, x2):
    num = np.abs(x1 - x2)
    den = np.abs(x1) + np.abs(x2)
    if den != 0:
        return num / den
    else:
        return 0


def gradient_check_sparse(f, W, grad, sample=10):
    """
    calculate the numerical gradient and compare with the input analytic gradient
    :param f: a function that takes weights W, data and label as input and output the loss, usually defined by lambda
    :param W: wights
    :param grad: analytic gradient obtained from back propagation
    :param sample: the number of sample place
    :return: numerical gradient
    """
    numerical_gradient_W = np.zeros_like(W)
    dW = 0.0001
    _, rows, cols, _ = W.shape
    for i in range(rows):
        for j in range(cols):
            delta_W = np.zeros_like(W)
            delta_W[0, i, j, 0] = dW
            numerical_gradient_W[0, i, j, 0] = (f(W + delta_W) - f(W)) / dW

    sample_indice = np.random.choice(np.arange(rows), sample)
    for i in sample_indice:
        print('numerical: %f, analytic: %f, relative error: %f' %
              (numerical_gradient_W[0, i, :, 0], grad[0, i, :, 0], relative_error(numerical_gradient_W[0, i, :, 0], grad[0, i, :, 0])))

    return numerical_gradient_W
