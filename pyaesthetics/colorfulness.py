#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains function to evaluate the colorfulness of an image in both the HSV and RGB color spaces.

Created on Mon Apr 16 11:49:45 2018
Last edited on Fri Aug 2 11:44:12 2024

@author: Giulio Gabrieli (gack94@gmail.com)
"""

import os #to handle filesystem files
import cv2 #for image manipulation
import numpy as np #numerical computation
import scipy.stats as stats

###############################################################################
#                                                                             #
#                              Colorfulness                                   #
#                                                                             #
###############################################################################

""" This section handles colorfulness estimation. """

import numpy as np
import cv2

def colorfulness_rgb(img):
    """ 
    Computes the colorfulness of an image using the metric described by Hasler and Suesstrunk (2003).

    :param img: Image to analyze, in RGB format.
    :type img: numpy.ndarray
    :return: Dictionary containing the colorfulness index and the mean R, G, B values.
    :rtype: dict
    """
    # Ensure the input is a NumPy array
    img = np.asarray(img)

    # Split the image into R, G, B components
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # Calculate mean R, G, B values
    mean_R, mean_G, mean_B = np.mean(R), np.mean(G), np.mean(B)
    std_R, std_G, std_B = np.std(R), np.std(G), np.std(B)

    # Evaluate rg and yb
    rg = R - G
    yb = 0.5 * (R + G) - B

    # Compute the standard deviation and mean of rgyb
    stdRGYB = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
    meanRGYB = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)

    # Compute the colorfulness index
    C = stdRGYB + 0.3 * meanRGYB

    return {
        "Colorfulness_RGB": C,
        "Mean_R": mean_R,
        "Mean_G": mean_G,
        "Mean_B": mean_B,
        "std_R":std_R,
        "std_G":std_G,
        "std_B":std_B
    }

def colorfulness_hsv(img, filter=False, saturation_low=0.2, value_low=0.15, value_high=0.95):
    """ 
    Computes the colorfulness of an image using the formula described in Yendrikhovskij et al., 1998.
    Input image is first converted to the HSV color space, then the S values are selected.

    Ci is evaluated with a sum of the mean S and its std, as in:
    
    Ci = mean(Si) + std(Si)

    :param img: Image to analyze, in RGB
    :type img: numpy.ndarray
    :return: Dictionary containing the colorfulness index and mean H, S, V values.
    :rtype: dict
    """
# Convert the image to the HSV color space
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Split the image into H, S, V components
    H, S, V = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]

    # Calculate mean H, S, V values
    mean_H, mean_S, mean_V = np.mean(H), np.mean(S), np.mean(V)
    std_H, std_S, std_V = np.std(H), np.std(S), np.std(V)

    # Evaluate the colorfulness index
    C = mean_S + np.std(S)

    # Initialize hsv variable
    hsv = img_hsv

    # Apply filtering if enabled
    if filter:
        mask = (img_hsv[:, 1] >= 255 * saturation_low) & (value_low * 255 <= img_hsv[:, 2]) & (img_hsv[:, 2] <= 255 * value_high)
        hsv = img_hsv[mask]

    # Return 0 if no valid pixels remain
    if hsv.size == 0:
        return {'color_variety': 0}

    # Flatten the hsv array for the bincount operation
    hsv_flat = hsv.reshape(-1, hsv.shape[-1])

    # Calculate color variety (entropy)
    counts = np.bincount(hsv_flat[:, 0] * 256**2 + hsv_flat[:, 1] * 256 + hsv_flat[:, 2])
    probs = counts[counts > 0] / hsv_flat.shape[0]
    color_variety = -np.sum(probs * np.log(probs))

    hue = H.flatten().astype(float)
    cm = stats.circmean(hue, high=180, low=0)
    cstd = stats.circstd(hue, high=180, low=0)

    return {
        "Colorfulness_HSV": C,
        "Mean_H": mean_H,
        "Mean_S": mean_S,
        "Mean_V": mean_V,
        "std_H": std_H,
        "std_S": std_S,
        "std_V": std_V,
        'color_variety': color_variety,
        "circular_mean_hue": cm,
        "circular_std_hue": cstd
    }


def object_count(img, lower_threshold=100, upper_threshold=200):
    # Load the image
    if img is None:
        print("Error: Image not found.")
        return 0

    # Apply Canny edge detection
    edges = cv2.Canny(img, lower_threshold, upper_threshold)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return len(contours)

###############################################################################
#                                                                             #
#                                  DEBUG                                      #
#                                                                             #
###############################################################################
        
""" For debug purposes."""

if __name__ == '__main__':
    
    basepath = os.path.dirname(os.path.realpath(__file__))

    # Path to a sample image for debugging   # Set the data path to use sample images
    data_folder = basepath + "/data/"
    
    # Path to a sample image
    sample_img = data_folder + "face1.png"
    # Read and convert the image
    img = cv2.imread(sample_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Print the colorfulness indices
    print(colorfulness_hsv(img))    
    print(colorfulness_rgb(img))
