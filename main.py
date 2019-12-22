#! /usr/bin/env python3

from browsers import browseImages
from descriptors import hu
from descriptors import zernike
from descriptors import surf
from descriptors import hough

train_images, numbers_labels = browseImages.numbersImages('./imgs/numbers/')
numbers_train_labels = sorted(numbers_labels * 7)


hu_moments_train = []
for number_images in train_images:
    for image in number_images:
        hu_moments_train.append(hu.hu_moments(image))

"""
zernike_moments_train = []
for number_images in train_images:
    number_zernike_moments = []
    for image in number_images:
        number_zernike_moments.append(zernike.zernike_moments(image))

    zernike_moments_train.append(number_zernike_moments)
"""
"""
surf_train = []
for number_images in train_images:
    number_surfs = []
    for image in number_images:
        number_surfs.append(surf.surf(image))
    surf_train.append(number_surfs)
"""
"""
hough_train = []
for number_images in train_images:
    number_houghs = []
    for image in number_images:
        number_houghs.append(hough.hough(image))
    hough_train.append(number_houghs)
"""
"""
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(zernike_moments_train, numbers_train_labels)
"""

"""
zernike_moments_train = []
for number_images in train_images:
    for image in number_images:
        zernike_moments_train.append(zernike.zernike_moments(image).tolist())
"""
from classifiers.randomForest import randomForest
import cv2 as cv

imtest = cv.imread("number.png", 0)
imtest_zernike = zernike.zernike_moments(imtest)

print(randomForest(hu_moments_train, numbers_train_labels, [imtest_zernike]))

