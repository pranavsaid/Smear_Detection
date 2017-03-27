from os import listdir
from os.path import isfile, join
import numpy
import cv2
import glob
import imutils
import os
from skimage.filters import threshold_adaptive
def detectSmear():
    cv_img = []
    sum_img = numpy.zeros((500, 500, 3), numpy.float)
    average_img = numpy.zeros((500, 500, 3), numpy.float)
    mask_img = numpy.zeros((500, 500, 1), numpy.float)

    for i in glob.glob("C:/Users/dpran/Desktop/Geo-spatial vision and visionary/sample_drive/sample/*.jpg"):
        n = cv2.imread(i)
        cv_img.append(n)
        n = imutils.resize(n, width=500, height=500)
        n = cv2.GaussianBlur(n, (3, 3), 0)
        img_array = numpy.array(n, dtype=numpy.float)
        sum_img = sum_img + img_array

    average_img = (average_img + sum_img) / len(cv_img)
    average_img = numpy.array(numpy.round(average_img), dtype=numpy.uint8)
    cv2.imwrite("Average.jpg", average_img)
    gray_image = cv2.cvtColor(average_img, cv2.COLOR_BGR2GRAY)
    threeshold_Img= threshold_adaptive(gray_image, 255, offset = 10)
    #ret,threeshold_Img=cv2.threshold(gray_image,127,255,cv2.THRESH_TOZERO)
    scaledImage = threeshold_Img.astype("uint8") * 255
    scaledImage = cv2.GaussianBlur(scaledImage, (3, 3), 0)
    edge_detected_image = cv2.Canny(scaledImage, 75, 200)

    im2, contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    list = []
    smeardetected = cv_img[0]
    smeardetected = imutils.resize(smeardetected, width=500, height=500)

    # Checking if the contour area is big enough to be called as smear
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        epsilon = 0.1*perimeter
        approximation = cv2.approxPolyDP(c, epsilon, True)
        (x, y), radius = cv2.minEnclosingCircle(c)
        if abs(cv2.contourArea(c) - 3.14 * radius ** 2) < 300 and cv2.contourArea(c) > 300:
            cv2.drawContours(smeardetected, [approximation], -1, (255, 255, 0), 2)
            cv2.drawContours(mask_img, [approximation], -1, (255, 255, 255), -1)
            list.append(c)
    cv2.imshow('smeardetected', cv2.WINDOW_NORMAL)
    cv2.imshow('smeardetected', smeardetected); cv2.waitKey(0)
    cv2.imshow('mask_img', cv2.WINDOW_NORMAL)
    cv2.imshow('mask_img', mask_img); cv2.waitKey(0)
      # hold windows open until user presses a key
    cv2.imwrite("mask.jpg", mask_img)
    cv2.imwrite("smear.jpg", smeardetected)
    return

if __name__ == '__main__':
    detectSmear()