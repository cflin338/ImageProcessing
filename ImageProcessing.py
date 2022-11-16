# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
#import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
import argparse
from PIL import Image

def DisplayInput(im):
    im.show()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default = 0, help = "Specify image file")
    parser.add_argument("--SmoothSize", default = 5, help = "Smooth kernel size")
    parser.add_argument("--SobelThreshold", default = 70, help = "[0,255]")
    return parser.parse_args()

def EdgeDetection(gray, threshold):
    #sobel operator
    #Gx = [[1, 0, -1], 
    #      [2, 0, -2],
    #      [1, 0, -1]] * A
    #Gy = [[1, 2, 1],
    #      [0, 0, 0],
    #      [-1, -2, -1]] * A
    
    #convolve 
    x = [[1, 0, -1], 
          [2, 0, -2],
          [1, 0, -1]]
    y = [[1, 2, 1],
          [0, 0, 0],
          [-1, -2, -1]]
    
    gx = ndimage.convolve(gray, x, mode = 'constant')
    gy = ndimage.convolve(gray, y, mode = 'constant')
    
    G = np.sqrt(gx*gx + gy*gy)
    #threshold = 70 #[0,255]
    #G = max(G, threshold)
    
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            if G[i,j] <= threshold:
                G[i,j] = 0
            else:
                G[i,j] = 255
    
    #plt.imshow(G)
    #plt.imshow(gx)
    #plt.show()
    return G


def ImageSmoothing(img, size, channels):
    conv = 1/np.product(size)*np.ones((size[0], size[1], channels))
    smooth = ndimage.convolve(img, conv, mode = 'constant')
    
    return smooth

def CartoonizeImage(img, width, height, arguments, channels):
    #1) transform into grayscale image .2989r + .587g + .114b
    #2) detect edges from grayscale
    #3) second copy, smooth colors
    #4) overlap images
    
    #weighted grayscale
    grayscale = .2989*img[:,:,0] + .587*img[:,:,1] + .114*img[:,:,2]
    
    NewImg = Image.fromarray(grayscale,)
    
    Edge = EdgeDetection(grayscale, arguments.SobelThreshold)
    NewImg = Image.fromarray(Edge)
    #NewImg.paste(grayscale)
    
    Smooth = ImageSmoothing(img, (arguments.SmoothSize,arguments.SmoothSize), channels)
    
    Image.fromarray(Smooth).show()
    
    return NewImg


if __name__=='__main__':
    arguments = parse_args()
    im = Image.open(arguments.image)
    #access inputs with command: arguments.image
    DisplayInput(im)
    
    #image object as a matrix:
    raw_img = np.asarray(im)
    (height, width, channels) = raw_img.shape
    
    cart = CartoonizeImage(raw_img, width, height, arguments, channels = 3)
    cart.show()
    
    im.close()
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    