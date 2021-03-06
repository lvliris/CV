from data_utils import load_CIFAR10
from k_nearest_neighbor import KNearestNeighbor
import matplotlib.pyplot as plt
import numpy as np


def time_function(f, *args):
    """call a function f with arguments args and return the time it takes"""
    import time
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic


if __name__ == '__main__':
    file = 'data/cifar-10-batches-py/'
    X_train, y_train, X_test, y_test = load_CIFAR10(file)
    mask = range(500)
    X_train = X_train[mask]
    y_train = y_train[mask]
    X_test = X_test[range(10)]
    y_test = y_test[range(10)]
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

    X_train = X_train.reshape(500, -1)
    y_train = y_train.reshape(500, -1)
    X_test = X_test.reshape(10, -1)
    y_test = y_test.reshape(10, -1)

    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)
    dists = classifier.compute_distance_two_loops(X_test)
    dists_one = classifier.compute_distance_one_loop(X_test)

    diff = np.linalg.norm(dists - dists_one, ord='fro')
    if diff < 0.001:
        print('good')
    else:
        print('bad')
    y_pred = classifier.predict_labels(dists, 1)
    correct = np.where(y_pred == y_train)
    print('accuracy: ', len(correct) / len(y_test))

    # cross validation
    num_folds = 5
    k_chioces = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

    X_train_folds = []
    y_train_folds = []
    X_train_folds = np.split(X_train, num_folds)
    y_train_folds = np.split(y_train, num_folds)

    # save the results
    k_to_accuracy = {}

    # find the best k by cross-validation
    classifier = KNearestNeighbor()
    for k in k_chioces:
        accuracies = []
        for fold in range(num_folds):
            X_temp = X_train_folds[:]
            y_temp = y_train_folds[:]

            X_validate_fold = X_temp.pop(fold)
            y_validate_fold = y_temp.pop(fold)

            X_temp = np.array([y for x in X_temp for y in x])
            y_temp = np.array([y for x in y_temp for y in x])

            classifier.train(X_temp, y_temp, k)

            y_validate_pred = classifier.predict_labels(X_validate_fold, k)
            num_correct = np.sum(y_validate_pred == y_validate_fold)
            accuracies[fold] = float(num_correct) / len(y_validate_fold)
        k_to_accuracy[k] = accuracies

    # print the accuracy
    for k in sorted(k_to_accuracy):
        for accuracy in k_to_accuracy[k]:
            print('k = %d, accuracy: %f' % (k, accuracy))

    # visualize the results
    for k in k_chioces:
        accuracies = k_to_accuracy[k]
        plt.scatter([k]*len(accuracies), accuracies)

    # plot the trend line with error bars that correspond to standard and mean
    accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracy)])
    accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracy)])
    plt.errorbar(k_chioces, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('accuracies')
    plt.show()

    # show the best classifier
    best_k = 10
    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)
    y_test_pred = classifier.predict_labels(X_test, k=best_k)
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / len(y_test)
    print('Got best accuracy: %f' % accuracy)
