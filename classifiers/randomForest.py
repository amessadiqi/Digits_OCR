#! /usr/bin/env python3

from sklearn.ensemble import RandomForestClassifier


def random_forest(train_vectors, train_labels_vectors, prediction_vector):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(train_vectors, train_labels_vectors)

    return clf.predict(prediction_vector)


def random_forest_recognition_rate(train_vectors, train_labels_vectors, prediction_vectors, prediction_vectors_labels):
    total = len(prediction_vectors)
    recognized = 0

    i = 0
    for vector in prediction_vectors:
        res = random_forest(train_vectors, train_labels_vectors, [vector])

        if res == prediction_vectors_labels[i]:
            recognized = recognized + 1

        i = i + 1

    return (recognized / total) * 100


if __name__ == "__main__":
    X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    y = ['one', 'two', 'three']

    rf = RandomForestClassifier(max_depth=2, random_state=0)
    rf.fit(X, y)

    print(rf.predict([[5, 6, 7]]))
