#! /usr/bin/env python3

from sklearn import svm


def svc(train_vectors, train_labels_vectors, prediction_vector):
    clf = svm.SVC()
    clf.fit(train_vectors, train_labels_vectors)

    return clf.predict(prediction_vector)


def svc_recognition_rate(train_vectors, train_labels_vectors, prediction_vectors, prediction_vectors_labels):
    total = len(prediction_vectors)
    recognized = 0

    i = 0
    for vector in prediction_vectors:
        res = svc(train_vectors, train_labels_vectors, [vector])

        if res == prediction_vectors_labels[i]:
            recognized = recognized + 1

        i = i + 1

    return (recognized / total) * 100


if __name__ == "__main__":
    X = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
    y = ['zero', 'one', 'two']

    clf = svm.SVC()
    clf.fit(X, y)

    print(clf.predict([[1, 1, 1]]))
