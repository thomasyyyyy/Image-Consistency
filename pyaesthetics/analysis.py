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
import gc

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
    from . import texture
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
    import texture

###############################################################################
#                                                                             #
#                      Quadratic Tree Decomposition                           #
#                                                                             #
###############################################################################
def get_image_dimensions(image):
    if image is not None:
        height, width = image.shape[:2]
        return (height, width)
    else:
        return (None, None)

def load_image(pathToImg):
    # Ensure the path is valid
    if not os.path.isfile(pathToImg):
        raise ValueError(f"File does not exist: {pathToImg}")

    # Handle PNG explicitly with PIL for better compatibility
    if pathToImg.lower().endswith(".png"):
        try:
            with Image.open(pathToImg) as pil_img:
                img_rgb = np.array(pil_img.convert('RGB'))
                img_bw = np.array(pil_img.convert('L'))
                return img_rgb, img_bw
        except Exception as e:
            raise ValueError(f"Failed to read PNG image: {e}")

    # Default handling for other formats
    img = cv2.imread(pathToImg)
    if img is None:
        raise ValueError(f"Image at {pathToImg} could not be read.")

    imageColor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imageBW = cv2.imread(pathToImg, 0)

    return imageColor, imageBW

def analyze_image(pathToImg, method='complete', sizing="resize", newSize=(200, 200), thumbnailSize=(400, 200), minStd=10, minSize=20):
    if method not in ['fast', 'complete']:
        raise ValueError(f'The specified method "{method}" is not supported. Method must be either fast or complete')

    resultdict = {}

    imageColor, imageBW = load_image(pathToImg)

    # Apply resizing or thumbnail logic
    if sizing == "resize":
        imageBW = cv2.resize(imageBW, newSize, interpolation=cv2.INTER_CUBIC)
        imageColor = cv2.resize(imageColor, newSize, interpolation=cv2.INTER_CUBIC)
    elif sizing == "thumbnail":
        with Image.open(pathToImg) as pil_img:
            pil_img.thumbnail(thumbnailSize, Image.LANCZOS)
            imageBW = np.array(pil_img.convert('L'))
            imageColor = np.array(pil_img.convert('RGB'))

    imgsRGB2RGB = utils.sRGB2RGB(imageColor)

    if method == 'fast':
        resultdict["brightness_BT709"] = brightness.relativeluminance_bt709(imgsRGB2RGB)
        resultdict["VC_quadTree"] = visualcomplexity.get_visual_complexity_quadtree(imageBW, minStd, minSize)
        resultdict["Symmetry_QTD"] = symmetry.get_symmetry(imageBW, minStd, minSize)
        resultdict["Colorfulness_RGB"] = colorfulness.colorfulness_rgb(imageColor)
        resultdict["contrast_RMS"] = contrast.contrast_rms(imageColor)
        resultdict["saturation"] = saturation.saturation(imageColor)

    elif method == 'complete':
        resultdict["brightness_BT709"] = brightness.relativeluminance_bt709(imgsRGB2RGB)
        resultdict["brightness_BT601"] = brightness.relativeluminance_bt601(imgsRGB2RGB)
        resultdict["VC_quadTree"] = visualcomplexity.get_visual_complexity_quadtree(imageBW, minStd, minSize)
        resultdict["VC_gradient"] = visualcomplexity.get_visual_complexity_gradient(imageColor)
        resultdict["VC_weight"] = visualcomplexity.get_visual_complexity_weight(pathToImg)
        resultdict["Symmetry_QTD"] = symmetry.get_symmetry(imageBW, minStd, minSize)
        resultdict["Colorfulness_HSV"] = colorfulness.colorfulness_hsv(imageColor)
        resultdict["Colorfulness_RGB"] = colorfulness.colorfulness_rgb(imageColor)
        resultdict["FacesCv2"] = facedetection.detect_faces_cv2(imageColor)
        resultdict["Number_of_Faces_Cv2"] = len(resultdict["FacesCv2"])
        resultdict["Colors"] = colordetection.get_colors_w3c(imageColor, ncolors=16)
        resultdict["contrast_rms"] = contrast.contrast_rms(imageColor)
        resultdict["contrast_michelson"] = contrast.contrast_michelson(imageColor)
        resultdict["saturation"] = saturation.saturation(imageColor)
        resultdict["shape"] = shape.attr_line_hough_edge(imageColor)
        resultdict["linesRatio"] = linesdetection.analyse_lines(imageColor)
        resultdict['selfSimilarity'] = selfsimilarity.get_self_similarity(imageColor)
        resultdict["sharpness"] = sharp.attr_sharp_laplacian(imageColor)
        resultdict["object_count"] = colorfulness.object_count(imageBW)
        resultdict["height"], resultdict["width"] = get_image_dimensions(imageColor)
        resultdict["texture"] = texture.main(imageBW)

    del imageColor, imageBW
    gc.collect()

    return resultdict

if __name__ == '__main__':
    basepath = os.path.dirname(os.path.realpath(__file__))

    data_folder = os.path.join(basepath, "data")

    sample_img = os.path.join(data_folder, "face1.png")

    start_time = time.time()
    results = analyze_image(sample_img, method='complete')
    end_time = time.time()

    print(utils.tablify_results(results).transpose())
    print(f"Execution time: {end_time - start_time:.4f} seconds")
