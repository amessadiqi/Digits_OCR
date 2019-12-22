#! /usr/bin/env python3

from mahotas.features import surf as sf


def surf(image):
    # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    desc = sf.surf(image)

    return desc


if __name__ == '__main__':
    import cv2 as cv
    image = cv.imread("../number.png")

    s = surf(image)
    print(s)
