from data_utils import get_CIFAR10_data
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2


def simple_model(X, y):
    """
    define a simple network with one convolution layer and one fully connected layer
    :param X: the input images, shape [N, 32, 32, 3]
    :param y: the label of the images, shape [N, 1]
    :return: the output of the model
    """
    # the weights of the convolution filters
    Wconv1 = tf.get_variable('Wconv1', dtype=tf.float32, shape=[7, 7, 3, 32])
    bconv1 = tf.get_variable('bconv1', dtype=tf.float32, shape=[32])

    # the weights of the fully connected layer
    W = tf.get_variable('W', dtype=tf.float32, shape=[5408, 10])
    b = tf.get_variable('b', dtype=tf.float32, shape=[10])

    # convolution layer
    z1 = tf.nn.conv2d(X, Wconv1, strides=[1, 2, 2, 1], padding='VALID') + bconv1
    # activation function
    a1 = tf.nn.relu(z1)
    # reshape the data
    reshaped_a1 = tf.reshape(a1, [-1, 5408])
    # fully connected layer
    y_out = tf.matmul(reshaped_a1, W) + b

    return y_out


def CBRP_module(input, i=1, filters=32):
    """
    Integrate the Convolution, BN, ReLU, Pooling layer
    :param i: the index of the module used to determinate names
    :param filters:
    :return:
    """
    # Conv layer
    channel = input.shape[3]
    Wconv1 = tf.get_variable('Wconv%d' % i, shape=[3, 3, channel, filters], dtype=tf.float32)
    bconv1 = tf.get_variable('bconv%d' % i, shape=[filters], dtype=tf.float32)
    h1 = tf.nn.conv2d(input, Wconv1, strides=[1, 1, 1, 1], padding='SAME') + bconv1

    # Spatial Batch Normalization layer
    axis = list(range(len(h1.get_shape()) - 1))
    mean, variance = tf.nn.moments(h1, axis)
    param_shape = h1.get_shape()[-1:]
    beta = tf.get_variable('beta%d' % i, shape=param_shape, initializer=tf.zeros_initializer)
    gamma = tf.get_variable('gamma%d' % i, shape=param_shape, initializer=tf.ones_initializer)

    # The mean and variance are saved as moving_mean and moving_var during training time
    # They are used as mean and variance in testing
    moving_mean = tf.get_variable('moving_mean%d' % i, shape=param_shape, initializer=tf.zeros_initializer, trainable=False)
    moving_var = tf.get_variable('moving_var%d' % i, shape=param_shape, initializer=tf.ones_initializer, trainable=False)

    # Update variable:= variable * decay + value * (1 - decay)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
    update_moving_var = moving_averages.assign_moving_average(moving_var, variance, BN_DECAY)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_var)

    mean, variance = control_flow_ops.cond(is_training, lambda: (mean, variance), lambda: (moving_mean, moving_var))
    h1_b = tf.nn.batch_normalization(h1, mean, variance, beta, gamma, BN_EPISILON)

    # ReLU activation layer
    a1 = tf.nn.relu(h1_b)

    # Max pooling layer
    p1 = tf.nn.max_pool(a1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    return p1


def complex_model(X, y):
    """
    define a complex model including several Conv and BN
    :param X: the input images, shape [N, 32, 32, 3]
    :param y: the label of the images, shape [N, 1]
    :return: the output of the model
    """
    filters = [32, 64, 128]
    p1 = X
    for i, filter_num in enumerate(filters):
        p1 = CBRP_module(p1, i, filters=filter_num)

    # Affine layer with 1024 output units
    p1_flat = tf.reshape(p1, shape=[-1, 131072])
    W1 = tf.get_variable('W1', shape=[131072, 1024], dtype=tf.float32)
    b1 = tf.get_variable('b1', shape=[1024], dtype=tf.float32)
    h2 = tf.matmul(p1_flat, W1) + b1

    # ReLU activation layer
    a2 = tf.nn.relu(h2)

    # Affine output layer
    W2 = tf.get_variable('W2', shape=[1024, 10], dtype=tf.float32)
    b2 = tf.get_variable('b2', shape=[10], dtype=tf.float32)
    y_out = tf.matmul(a2, W2) + b2

    return y_out


def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # shuffle indicies
    num_train = Xd.shape[0]
    train_indicies = np.arange(num_train)
    np.random.shuffle(train_indicies)

    training_now = training is not None

    # setting up variables we want to compute and optimize
    # if we have a training function, add that to things we have
    variables = [mean_loss, correct_prediction, accuracy]
    if training_now:
        variables[-1] = training

    iter = 0
    total_loss, total_correct = 0.0, 0.0
    losses = []
    for e in range(epochs):
        # keep track loss and accuracy
        correct = 0
        for i in range(num_train//batch_size):
            start_idx = i * batch_size
            idx = train_indicies[start_idx:start_idx+batch_size]

            # create a dictionary fro training
            feed_dict = {X: Xd[idx],
                         y: yd[idx],
                         is_training: training_now}

            # have tensorflow compute loss and correct predicitons
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables, feed_dict=feed_dict)

            # aggregate performance stats
            losses.append(loss*batch_size)
            correct += np.sum(corr)

            # print the training process
            if training_now and iter % print_every == 0:
                print('Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2: .2g}'\
                      .format(iter, loss, np.sum(corr)/batch_size))
            iter += 1

        total_correct = correct/num_train
        total_loss = np.sum(losses)/num_train
        print('Epoch {2}, overall loss = {0:.3g} and accuracy of {1:.3g}'\
              .format(total_loss, total_correct, e+1))

    if plot_losses:
        plt.plot(losses)
        plt.grid(True)
        plt.title('training losses')
        plt.xlabel('iteration number')
        plt.ylabel('loss')
        plt.show()

    return total_loss, total_correct


if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data(num_training=49000,
                                                                      num_validation=1000,
                                                                      num_test=1000)
    # convert to gray image
    X_train_gray = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in X_train]).reshape([-1, 32, 32, 1])
    X_val_gray = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in X_val]).reshape([-1, 32, 32, 1])
    X_test_gray = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in X_test]).reshape([-1, 32, 32, 1])

    # subtract the mean value
    mean_img = np.mean(X_train, axis=0).astype(np.float32)
    X_train = X_train.astype(np.float32) - mean_img
    X_val = X_val.astype(np.float32) - mean_img
    X_test = X_test.astype(np.float32) - mean_img
    mean_img_gray = np.mean(X_train_gray, axis=0).astype(np.float32)
    X_train_gray = X_train_gray.astype(np.float32) - mean_img_gray
    X_val_gray = X_val_gray.astype(np.float32) - mean_img_gray
    X_test_gray = X_test_gray.astype(np.float32) - mean_img_gray

    '''img = cv2.resize(X_train_gray[1000], (320, 320))
    cv2.imshow("img", img)
    cv2.waitKey(100)
    color_img = cv2.resize(X_train[1000], (320, 320))
    cv2.imshow("color img", color_img)
    cv2.waitKey(100)'''

    # clear old variables
    tf.reset_default_graph()

    # setup inputs
    X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='X')
    y = tf.placeholder(tf.int64, shape=[None], name='label')
    is_training = tf.placeholder(tf.bool)

    MOVING_AVERAGE_DECAY = 0.9997
    BN_DECAY = MOVING_AVERAGE_DECAY
    BN_EPISILON = 0.001

    y_out = complex_model(X, y)
    total_loss = tf.losses.hinge_loss(tf.one_hot(y, 10), logits=y_out)
    mean_loss = tf.reduce_mean(total_loss)

    optimizer = tf.train.AdamOptimizer(5e-4)
    train_step = optimizer.minimize(mean_loss)

    with tf.Session() as sess:
        with tf.device("/gpu:0"):   # /cpu:0 or /gpu:0
            sess.run(tf.global_variables_initializer())
            print('training')
            run_model(sess, y_out, mean_loss, X_train, y_train, 10, 64, 100, train_step, True)
            print('validation')
            run_model(sess, y_out, mean_loss, X_test, y_test, 1, 64)
