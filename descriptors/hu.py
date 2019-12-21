#! /usr/bin/env python3

import cv2 as cv
from math import *


def hu_moments(img):
    moments = cv.moments(img)
    hu_moments = cv.HuMoments(moments)

    for i in range(0, 7):
        hu_moments[i] = -1 * copysign(1.0, hu_moments[i]) * log10(abs(hu_moments[i]))

    return hu_moments


if __name__ == "__main__":
    import cv2 as cv

    image = cv.imread("../1.png", 0)

    moments = hu_moments(image)
    print(moments)
