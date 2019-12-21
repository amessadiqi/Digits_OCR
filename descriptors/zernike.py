#! /usr/bin/env python3

import mahotas as mhs


def zernike_moments(image):
    moments = mhs.features.zernike_moments(image, 21)

    return moments


if __name__ == "__main__":
    import cv2 as cv
    image = cv.imread("../1.png", 0)

    moments = zernike_moments(image)
    print(moments)
