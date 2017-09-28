# -*- coding: utf-8 -*-
"""
Created on Wed May 24 16:34:56 2017

@author: KRapes
"""
import cv2
from matplotlib import pyplot as plt
from skimage import img_as_ubyte
import numpy as np
plt.ion()


def show_img(i):
    #i = img_as_ubyte(i)

    #i = img_as_ubyte(i)
    cv2.imshow('image',i)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #plt.close('all')
    
def rotate_img(img, degrees):
    height, width = img.shape
    M = cv2.getRotationMatrix2D((width/2, height/2),degrees,1)
    dst = cv2.warpAffine(img,M,(width,height))
    return dst

def blur(img):
    blurred = cv2.GaussianBlur(img, (3,3), 0)
    return blurred

def Canny_Edge_Detection(img, sigma=0.33):
    #Threshold1 = 25;
    #Threshold2 = 100;
    Filtersize = 10
    #sigma = 0.33
    med = np.median(img)
    Threshold1 = int(max(0, (1.0 - sigma) * med))
    Threshold2 = int(min(255, (1.0 + sigma) * med))

        
    E = cv2.Canny(img, Threshold1, Threshold2, Filtersize)
    return E
    
img = cv2.imread("images/IMG_20170524_154436.jpg",0)
plt.close('all')

    # Canny Edge Detection:
plt.figure('0.2')
plt.imshow(Canny_Edge_Detection(blur(img), 0.2), cmap='gray')

plt.figure('0.33')
plt.imshow(Canny_Edge_Detection(blur(img), 0.33), cmap='gray')

plt.figure('0.5')
plt.imshow(Canny_Edge_Detection(blur(img), 0.5), cmap='gray')

plt.figure('0.66')
plt.imshow(Canny_Edge_Detection(blur(img), 0.66), cmap='gray')

plt.figure('0.75')
plt.imshow(Canny_Edge_Detection(blur(img), 0.75), cmap='gray')

plt.figure('0.85')
plt.imshow(Canny_Edge_Detection(blur(img), 0.85), cmap='gray')

plt.figure('0.95')
plt.imshow(Canny_Edge_Detection(blur(img), 0.95), cmap='gray')
#E = rotate_img(E, 0)
#E = img_as_ubyte(E)
#plt.close('all')
#plt.imshow(E, cmap = 'gray')
plt.show
show_img(img)