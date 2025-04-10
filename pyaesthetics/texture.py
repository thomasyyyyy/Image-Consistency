import numpy as np
import cv2
import math

import cv2 as cv
import numpy as np
import math

def coarseness(img):
    r, c = img.shape
    img = np.float32(img)

    A1 = np.zeros((r,c))
    A2 = np.zeros((r,c))
    A3 = np.zeros((r,c))
    A4 = np.zeros((r,c))
    A5 = np.zeros((r,c))
    A6 = np.zeros((r,c))

    Sbest = np.zeros((r, c))

    E1h = np.zeros((r, c))
    E1v = np.zeros((r, c))
    E2h = np.zeros((r, c))
    E2v = np.zeros((r, c))
    E3h = np.zeros((r, c))
    E3v = np.zeros((r, c))
    E4h = np.zeros((r, c))
    E4v = np.zeros((r, c))
    E5h = np.zeros((r, c))
    E5v = np.zeros((r, c))
    E6h = np.zeros((r, c))
    E6v = np.zeros((r, c))

    flag = 0

    for x in range(1, r):
        for y in range(1, c):
            A1[x,y] = (np.sum(np.sum(img[x-1:x+1,y-1:y+1])))

    for x in range(1, r - 1):
        for y in range(1, c - 1):
            E1h[x,y] = A1[x+1,y]-A1[x-1,y]
            E1v[x,y] = A1[x,y+1]-A1[x,y-1]

    E1h = E1h/2**(2*1)
    E1v = E1v/2**(2*1)

    if r < 4 or c < 4:
        flag = 1
    
    if flag == 0:
        for x in range(2, r - 1):
            for y in range(2, c - 1):
                A2[x,y]=(np.sum(np.sum(img[x-2:x+2, y-2:y+2])))
        
        for x in range(2, r - 2):
            for y in range(2, c - 2):
                E2h[x,y] = A2[x+2,y]-A2[x-2,y]
                E2v[x,y] = A2[x,y+2]-A2[x,y-2]

    E2h = E2h/2**(2*2)
    E2v = E2v/2**(2*2)

    if r < 8 or c < 8:
        flag = 1
    
    if flag == 0:
        for x in range(4, r - 3):
            for y in range(4, c - 3):
                A3[x,y]=(np.sum(np.sum(img[x-4:x+4, y-4:y+4])))
        
        for x in range(4, r - 4):
            for y in range(4, c - 4):
                E3h[x,y] = A3[x+4,y]-A3[x-4,y]
                E3v[x,y] = A3[x,y+4]-A3[x,y-4]

    E3h = E3h/2**(2*3)
    E3v = E3v/2**(2*3)

    if r < 16 or c < 16:
        flag = 1
    
    if flag == 0:
        for x in range(8, r - 7):
            for y in range(8, c - 7):
                A4[x,y]=(np.sum(np.sum(img[x-8:x+8, y-8:y+8])))
        
        for x in range(8, r - 8):
            for y in range(8, c - 8):
                E4h[x,y] = A4[x+8,y]-A4[x-8,y]
                E4v[x,y] = A4[x,y+8]-A4[x,y-8]

    E4h = E4h/2**(2*4)
    E4v = E4v/2**(2*4)

    if r < 32 or c < 32:
        flag = 1
    
    if flag == 0:
        for x in range(16, r - 15):
            for y in range(16, c - 15):
                A5[x,y]=(np.sum(np.sum(img[x-16:x+16, y-16:y+16])))
        
        for x in range(16, r - 16):
            for y in range(16, c - 16):
                E5h[x,y] = A5[x+16,y]-A5[x-16,y]
                E5v[x,y] = A5[x,y+16]-A5[x,y-16]

    E5h = E5h/2**(2*5)
    E5v = E5v/2**(2*5)

    if r < 64 or c < 64:
        flag = 1
    
    if flag == 0:
        for x in range(64, r - 63):
            for y in range(64, c - 63):
                A6[x,y]=(np.sum(np.sum(img[x-64:x+64, y-64:y+64])))
        
        for x in range(64, r - 64):
            for y in range(64, c - 64):
                E6h[x,y] = A6[x+64,y]-A6[x-64,y]
                E6v[x,y] = A6[x,y+64]-A6[x,y-64]

    E6h = E6h/2**(2*6)
    E6v = E6v/2**(2*6)

    for i in range(0, r):
        for j in range(0, c):
            maxv = np.max(np.array([abs(E1h[i,j]), abs(E1v[i,j]), abs(E2h[i,j]), abs(E2v[i,j]), abs(E3h[i,j]), abs(E3v[i,j]), abs(E4h[i,j]), abs(E4v[i,j]), abs(E5h[i,j]), abs(E5v[i,j]), abs(E6h[i,j]), abs(E6v[i,j])]))
            index = np.argmax(np.array([abs(E1h[i,j]), abs(E1v[i,j]), abs(E2h[i,j]), abs(E2v[i,j]), abs(E3h[i,j]), abs(E3v[i,j]), abs(E4h[i,j]), abs(E4v[i,j]), abs(E5h[i,j]), abs(E5v[i,j]), abs(E6h[i,j]), abs(E6v[i,j])]))
            k=math.floor((index+1)/2)
            Sbest[i, j] = 2 ** k

    return np.sum(np.sum(Sbest))/(r*c)

def contrast(img):
    img = np.float32(img)
    M4 = np.mean((img - np.mean(img)) ** 4)
    delta2 = np.var(img)
    alfa4 = M4 / (delta2 ** 2)
    delta = np.std(img)
    return delta / (alfa4 ** (1 / 4))

def directionality(image):
    h, w = image.shape
    convH = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    convV = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    # Convolve with kernels
    deltaH = cv2.filter2D(image, -1, convH)
    deltaV = cv2.filter2D(image, -1, convV)

    deltaG = (np.abs(deltaH) + np.abs(deltaV)) / 2.0
    deltaG_vec = deltaG.flatten()

    theta = np.arctan2(deltaV, deltaH) + np.pi / 2
    theta[deltaH == 0] = np.pi / 2
    theta_vec = theta.flatten()

    # Create the histogram of directions
    n = 16
    t = 12
    hd = np.zeros(n)
    for ni in range(n):
        hd[ni] = np.sum((deltaG_vec >= t) & (theta_vec >= (2 * ni - 1) * np.pi / (2 * n)) & (theta_vec < (2 * ni + 1) * np.pi / (2 * n)))

    hd /= np.mean(hd)
    hd_max_index = np.argmax(hd)

    # Calculate final directionality score
    fdir = np.sum(np.square(np.arange(n) - hd_max_index) * hd)
    return fdir

def main(image):
    # Check if the image was loaded successfully
    if image is None:
        print("Error: Could not load image.")
        return

    # Calculate the image features
    coarseness_value = coarseness(image)
    contrast_value = contrast(image)
    directionality_value = directionality(image)
    
    return {"coarseness": coarseness_value, "contrast": contrast_value, "directionality": directionality_value}

# Example usage
if __name__ == "__main__":
    image_path = "C:/Users/txtbn/OneDrive/Pictures/Exploring.jpg"  # Replace with the path to your image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (200, 200))
    
    main(image)
