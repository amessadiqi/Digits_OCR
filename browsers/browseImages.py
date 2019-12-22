#! /usr/bin/env python3

import os
import cv2 as cv


def numbers_images(path):
    images = []
    numbers_label = os.listdir(path)

    for number_img in numbers_label:
        img_files = os.listdir(path + number_img)
        folder_images = []
        for img_file in img_files:
            img_path = path + number_img + '/' + img_file
            image = cv.imread(img_path)

            folder_images.append(image)

        images.append(folder_images)

    return images, numbers_label


if __name__ == '__main__':
    images, labels = numbersImages('../imgs/numbers/')
    print(images)
    print(labels)
