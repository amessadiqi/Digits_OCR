#! /usr/bin/env python3

import classifiers.knn as knn
import classifiers.svm as svm
import classifiers.randomForest as randf

from browsers import browseImages
from descriptors import hu
from descriptors import zernike


# Preparing learning images for training
train_images, numbers_labels = browseImages.numbersImages('./imgs/numbers/')
numbers_train_labels = sorted(numbers_labels * len(train_images[0]))

# Loading demo images
demo_images, demo_numbers_labels = browseImages.numbersImages('./imgs/demo/')
demo_numbers_labels = sorted(demo_numbers_labels * len(demo_images[0]))

# Classification using Hu moments
# Step 1: Calculating Hu moments for all database images and demo images

print("Hu Moments : ")

hu_train_vectors = hu.multiple_hu_moments(train_images)
hu_train_demo_vectors = hu.multiple_hu_moments(demo_images)

# Step 2: Classifying images using KNN

rate = knn.knn_recognition_rate(hu_train_vectors, numbers_train_labels,
                                hu_train_demo_vectors, demo_numbers_labels)
print("KNearest-neighbors recognition rate:", rate, "%")

rate = svm.svc_recognition_rate(hu_train_vectors, numbers_train_labels,
                                hu_train_demo_vectors, demo_numbers_labels)
print("Svc recognition rate:", rate, "%")

rate = randf.random_forest_recognition_rate(hu_train_vectors, numbers_train_labels,
                                hu_train_demo_vectors, demo_numbers_labels)
print("Random forest recognition rate:", rate, "%")

# Classification using Zernike moments
# Step 1: Calculating Zernike moments for all database images and demo images

print("\n\nZernike Moments : ")

zernike_train_vectors = zernike.multiple_zernike_moments(train_images)
zernike_train_demo_vectors = zernike.multiple_zernike_moments(demo_images)

# Step 2: Classifying images using KNN

rate = knn.knn_recognition_rate(zernike_train_vectors, numbers_train_labels,
                                zernike_train_demo_vectors, demo_numbers_labels)
print("KNearest-neighbors recognition rate:", rate, "%")

rate = svm.svc_recognition_rate(zernike_train_vectors, numbers_train_labels,
                                zernike_train_demo_vectors, demo_numbers_labels)
print("Svc recognition rate:", rate, "%")

rate = randf.random_forest_recognition_rate(zernike_train_vectors, numbers_train_labels,
                                zernike_train_demo_vectors, demo_numbers_labels)
print("Random forest recognition rate:", rate, "%")
