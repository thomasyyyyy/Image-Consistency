#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is an entrypoint for the automatic analysis of images using pyaesthetics.


Created on Mon Apr 16 22:40:46 2018
Last edited on Fri Sep 13 16:30:00 2024

@author: Giulio Gabrieli (gack94@gmail.com)
"""

###############################################################################
#                                                                             #
#                                 Libraries                                   #
#                                                                             #
###############################################################################

import os
import cv2  # OpenCV library for image processing
import matplotlib.pyplot as plt  # Matplotlib for plotting images
import pytesseract  # Pytesseract for Optical Character Recognition (OCR)
from PIL import Image  # Python Imaging Library for image processing
import pandas as pd #for pandas tables
from tqdm import tqdm
import numpy as np
import time

# Attempt to import internal modules of pyaesthetics, handling both relative and absolute imports
try:
    from . import brightness
    from . import colordetection
    from . import colorfulness
    from . import contrast
    from . import facedetection
    from . import linesdetection
    from . import quadtreedecomposition
    from . import saturation
    from . import selfsimilarity
    from . import spacebaseddecomposition
    from . import symmetry
    from . import utils
    from . import visualcomplexity
    from . import sharp
    from . import shape
except:
    import brightness1 as brightness
    import colordetection
    import colorfulness
    import contrast
    import facedetection
    import linesdetection
    import quadtreedecomposition
    import saturation
    import selfsimilarity
    import spacebaseddecomposition
    import symmetry
    import utils
    import visualcomplexity
    import sharp
    import shape

###############################################################################
#                                                                             #
#                      Quadratic Tree Decomposition                           #
#                                                                             #
###############################################################################


def get_image_dimensions(image):
    return image.shape[:2] if image is not None else (None, None)

def analyze_image(pathToImg, method='fast', resize=True, newSize=(600, 400), minStd=10, minSize=20):
    """Analyze an image's aesthetic features using either 'fast' or 'complete' methods."""
    if method not in ['fast', 'complete']:
        raise ValueError(f'Invalid method: {method}. Choose either "fast" or "complete".')

    resultdict = {}

    img = cv2.imread(pathToImg)
    if img is None:
        raise FileNotFoundError(f"Image not found: {pathToImg}")

    imageColor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imageBW = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if resize:
        imageBW = cv2.resize(imageBW, newSize, interpolation=cv2.INTER_CUBIC)
        imageColor = cv2.resize(imageColor, newSize, interpolation=cv2.INTER_CUBIC)

    imgsRGB2RGB = utils.sRGB2RGB(img)

    # Core analysis features
    resultdict.update({
        "brightness_BT709": brightness.relativeluminance_bt709(imgsRGB2RGB),
        "VC_quadTree": visualcomplexity.get_visual_complexity_quadtree(imageBW, minStd, minSize),
        "Symmetry_QTD": symmetry.get_symmetry(imageBW, minStd, minSize),
        "Colorfulness_RGB": colorfulness.colorfulness_rgb(imageColor),
        "contrast_RMS": contrast.contrast_rms(imageColor),
        "saturation": saturation.saturation(imageColor),
        "height": img.shape[0],
        "width": img.shape[1]
    })

    if method == 'complete':
        resultdict.update({
            "brightness_BT601": brightness.relativeluminance_bt601(imgsRGB2RGB),
            "VC_gradient": visualcomplexity.get_visual_complexity_gradient(imageColor),
            "VC_weight": visualcomplexity.get_visual_complexity_weight(pathToImg),
            "Colorfulness_HSV": colorfulness.colorfulness_hsv(imageColor),
            "FacesCv2": facedetection.detect_faces_cv2(imageColor),
            "Number_of_Faces_Cv2": len(facedetection.detect_faces_cv2(imageColor)),
            "Colors": colordetection.get_colors_w3c(imageColor, ncolors=16),
            "contrast_michelson": contrast.contrast_michelson(imageColor),
            "shape": shape.attr_line_hough_edge(img),
            "linesRatio": linesdetection.analyse_lines(imageColor),
            "selfSimilarity": selfsimilarity.get_self_similarity(img),
            "sharpness": sharp.attr_sharp_laplacian(img),
            "object_count": colorfulness.object_count(imageBW)
        })

    return resultdict

if __name__ == '__main__':
    basepath = os.path.dirname(os.path.realpath(__file__))
    sample_img = os.path.join(basepath, "data", "face1.png")

    start_time = time.time()
    results = analyze_image(sample_img, method='complete')
    end_time = time.time()

    print(utils.tablify_results(results).transpose())
    print(f"Execution time: {end_time - start_time:.4f} seconds")