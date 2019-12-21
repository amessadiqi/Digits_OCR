#! /usr/bin/env python3

from mahotas.features import surf as sf


def surf(image):
    desc = sf.surf(image)

    return desc


if __name__ == '__main__':
    import cv2 as cv
    image = cv.imread("../1.png", 0)

    s = surf(image)
    print(s)
