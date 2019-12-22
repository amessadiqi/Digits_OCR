#! /usr/bin/env python3

from sklearn import svm


def SVC(train_vectors, train_labels_vectors, prediction_vector):
    svc = svm.SVC()
    svc.fit(train_vectors, train_labels_vectors)

    return svc.predict(prediction_vector)


if __name__ == "__main__":
    X = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
    y = ['zero', 'one', 'two']

    clf = svm.SVC()
    clf.fit(X, y)

    print(clf.predict([[1, 1, 1]]))
