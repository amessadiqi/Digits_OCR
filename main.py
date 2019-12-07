import cv2 as cv
import descriptors.hu as hu

img = cv.imread("imgs/numbers/one/1.png", cv.IMREAD_GRAYSCALE)
imgbis = cv.imread("imgs/numbers/one/1bis.png", cv.IMREAD_GRAYSCALE)

_, img = cv.threshold(img, 128, 255, cv.THRESH_BINARY)
_, imgbis = cv.threshold(imgbis, 128, 255, cv.THRESH_BINARY)

print(hu.hu(img))
print(hu.hu(imgbis))
