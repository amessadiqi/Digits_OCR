#! /usr/bin/env python3

from sklearn.ensemble import RandomForestClassifier


def randomForest(train_vectors, train_labels_vectors, prediction_vector):
    rf = RandomForestClassifier(max_depth=2, random_state=0)
    rf.fit(train_vectors, train_labels_vectors)

    return rf.predict(prediction_vector)


if __name__ == "__main__":
    X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    y = ['one', 'two', 'three']

    rf = RandomForestClassifier(max_depth=2, random_state=0)
    rf.fit(X, y)

    print(rf.predict([[5, 6, 7]]))
