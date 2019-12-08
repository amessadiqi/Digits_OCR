#! /usr/bin/env python3

import cv2 as cv


def closestNeighbor(learn_imgs, input_img):
    distances = []

    for img in learn_imgs:
        distances.append(cv.matchShapes(img, input_img, cv.CONTOURS_MATCH_I1, 0))

    return learn_imgs[distances.index(min(distances))]


if __name__ == "__main__":
    pass
