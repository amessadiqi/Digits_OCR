#! /usr/bin/env python3

from sklearn.neighbors import KNeighborsClassifier


def KNN(train_vectors, train_labels_vectors, prediction_vector):
    knn = KNeighborsClassifier(n_neighbors=9)
    knn.fit(train_vectors, train_labels_vectors)

    return knn.predict(prediction_vector)


if __name__ == "__main__":
    X = [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], [[0.1, 0.1, 0.1], [0.11, 0.11, 0.11], [0.12, 0.12, 0.12]], [[0.2, 0.2, 0.2], [0.21, 0.21, 0.21], [0.22, 0.22, 0.22]], [[0.3, 0.3, 0.3], [0.31, 0.31, 0.31], [0.32, 0.32, 0.32]]]
    y = ['one', 'one', 'two', 'two']

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    result = knn.predict([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]])

    print(result)