#! /usr/bin/env python3

import cv2 as cv
import numpy as np


def hough(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    min_line_length = 100
    max_line_gap = 10
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, min_line_length, max_line_gap)

    return lines


if __name__ == '__main__':
    img = cv.imread('../number.png')

    print(hough(img))
