#! /usr/bin/env python3

from mahotas.features import zernike_moments as zm
import cv2 as cv


def zernike_moments(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    moments = zm(gray_image, 21)

    return moments


def multiple_zernike_moments(images):
    zernike_moments_all = []
    for element in images:
        for image in element:
            zernike_moments_all.append(zernike_moments(image).tolist())

    return zernike_moments_all


if __name__ == "__main__":
    import cv2 as cv
    image = cv.imread("../number.png", 0)

    moments = zernike_moments(image)
    print(moments)
