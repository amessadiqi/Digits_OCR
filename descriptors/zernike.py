#! /usr/bin/env python3

import mahotas as mhs


def zernike_moments(img):
    moments = mhs.features.zernike_moments(img, 1)

    return moments


if __name__ == "__main__":
    import cv2 as cv
    img = cv.imread("../img/numbers/one/1.png", 0)
    print(zernike_moments(img))
