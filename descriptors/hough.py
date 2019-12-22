#! /usr/bin/env python3

import cv2 as cv
import numpy as np


def hough(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    min_line_length = 100
    max_line_gap = 10
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, min_line_length, max_line_gap)

    res = []
    lines = list(lines)
    for line in lines:
        for element in line:
            for i in element:
                res.append(i)

    return res


def multiple_hough(images):
    hough_all = []
    for element in images:
        for image in element:
            hough_all.append(hough(image))

    return hough_all


if __name__ == '__main__':
    img = cv.imread('../number.png')

    print(hough(img))
