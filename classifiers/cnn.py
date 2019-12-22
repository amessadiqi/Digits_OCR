#! /usr/bin/env python3

from sklearn.neural_network import MLPClassifier


def mlp(train_vectors, train_labels_vectors, prediction_vector):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(train_vectors, train_labels_vectors)

    return clf.predict(prediction_vector)


def mlp_recognition_rate(train_vectors, train_labels_vectors, prediction_vectors, prediction_vectors_labels):
    total = len(prediction_vectors)
    recognized = 0

    i = 0
    for vector in prediction_vectors:
        res = mlp(train_vectors, train_labels_vectors, [vector])

        if res == prediction_vectors_labels[i]:
            recognized = recognized + 1

        i = i + 1

    return (recognized / total) * 100

if __name__ == '__main__':
    pass
