#!/usr/bin/env python
# file for comparing two images passed in by relative path

import numpy as np
from PIL import Image

def compare_imgs(img1, img2):
    '''function gets the sum of
    the difference of the grayscale values of two images
    and returns said error'''

    gray1 = Image.open(img1,'r').convert('L')  # open image and make grayscale
    gray2 = Image.open(img2,'r').convert('L')
    diff = np.subtract(gray1,gray2)  # get difference between matrices
    absfunc = np.vectorize(abs) 
    absdiff = absfunc(diff) # get absolute differences
    return absdiff.sum()

print(str(compare_imgs('test.png','test2.png')))
