#! /usr/bin/env python3

import os
import cv2 as cv
import matplotlib.pyplot as plt
from classifiers import matchShapes as ms

learn_imgs = []

path = 'imgs/numbers/'
numbers_label = os.listdir(path)

for number_img in numbers_label:
    img_files = os.listdir(path + number_img)
    for img_file in img_files:
        img_path = path + number_img + '/' + img_file

        img = cv.imread(img_path, 0)
        img = cv.bitwise_not(img)

        learn_imgs.append(img)

rec_total = 0
total = 0

path = 'imgs/demo/'
demo_numbers_label = os.listdir(path)

for number_img in demo_numbers_label:
    img_files = os.listdir(path + number_img)
    for img_file in img_files:
        img_path = path + number_img + '/' + img_file

        input_img = cv.imread(img_path, 0)
        input_img = cv.bitwise_not(input_img)

        output_img = ms.closestNeighbor(learn_imgs, input_img)

        plt.subplot(1, 2, 1)
        plt.imshow(input_img)
        plt.subplot(1, 2, 2)
        plt.imshow(output_img)
        plt.show()

        res = int(input("Do the two images match ?\n"))
        if res == 1:
            rec_total = rec_total + 1
        total = total + 1

print("Recognised :", rec_total)
print("Total :", total)
print("Recognition rate :", (rec_total/total)*100, "%")

"""
elements = os.walk(path)

for files in elements:
    for file in files[2]:
        if file[-3:] == 'png':  # You can modify this condition depending on your supported formats
            img = im.imread(files[0] + '/' + file)

            plt.imshow(img)
            plt.show()
"""
