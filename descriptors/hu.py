#! /usr/bin/env python3

import cv2 as cv
from math import *


def hu_moments(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    moments = cv.moments(gray_image)
    hu_moments = cv.HuMoments(moments)

    for i in range(0, 7):
        hu_moments[i] = -1 * copysign(1.0, hu_moments[i]) * log10(abs(hu_moments[i]))

    hu_moments_adapted = []
    for element in hu_moments:
        hu_moments_adapted.append(float(element))

    return hu_moments_adapted


def multiple_hu_moments(images):
    hu_moments_all = []
    for element in images:
        for image in element:
            hu_moments_all.append(hu_moments(image))

    return hu_moments_all


if __name__ == "__main__":
    import cv2 as cv

    image = cv.imread("../number.png", 0)

    moments = hu_moments(image)
    print(moments)
