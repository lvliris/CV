import numpy as np


class KNearestNeighbor():
    def train(self, x, y):
        # todo: realize knn training
        self.X_train  = x
        self.Y_train = y

    def compute_distance_two_loops(self, x):
        # realize distance computation with two loops
        num_train = self.X_train.shape[0]
        num_test = x.shape[0]
        dist = np.zeros([num_test, num_train])

        for i in range(num_test):
            for j in range(num_train):
                dist[i, j] = np.sqrt(np.sum(np.square(x[i] - self.X_train[j])))

        return dist

    def compute_distance_one_loop(self, x):
        # todo: realize distance computation with one loop
        num_train = self.X_train.shape[0]
        num_test = x.shape[0]
        dist = np.zeros([num_test, num_train])

        for i in range(num_test):
            dist[i, :] = np.sqrt(np.sum(np.square(self.X_train - x[i, :]), axis=1))

        return dist

    def compute_distance_no_loops(self, x):
        # todo: realize distance computation with no loops
        pass

    def predict_labels(self, dists, k=1):
        # realize labels prediction with k nearest neighbor
        num_test = dists.shape[0]
        pred = np.zeros(num_test)

        for i in range(num_test):
            nearest_index = []
            nearest_index = self.Y_train[np.argsort(dists[i])[:k]]
            pred[i] = np.argmax(np.bincount(nearest_index))

        return pred

