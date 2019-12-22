#! /usr/bin/env python3

from mahotas.features import surf as sf


def surf(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    desc = sf.surf(gray_image)

    surf_vector = []

    for vector in desc:
        for i in vector:
            surf_vector.append(i)

    return surf_vector


def multiple_surf(images):
    surf_all = []
    for element in images:
        for image in element:
            surf_all.append(surf(image))

    return surf_all


if __name__ == '__main__':
    import cv2 as cv
    image = cv.imread("../number.png")

    s = surf(image)
    print(s)
