import numpy as np

def relativeluminance_bt709(img):
    """ 
    This function evaluates the brightness of an image by means of Y, where Y is evaluated as:
            
    Y = 0.7152G + 0.0722B + 0.2126R
    B = mean(Y)
        
    :param img: image to analyze, in RGB
    :type img: numpy.ndarray
    :return: mean brightness
    :rtype: float
    """
    # Ensure the image is in RGB format (using numpy arrays)
    img = np.array(img)
    
    # Vectorized calculation of brightness using the BT.709 formula
    B = np.mean(np.dot(img[..., :3], [0.2126, 0.7152, 0.0722]))  # Dot product for each pixel
    
    return B  # Return the mean brightness value

def relativeluminance_bt601(img):
    """ 
    This function evaluates the brightness of an image by means of Y, where Y is evaluated as:
            
    Y = 0.587G + 0.114B + 0.299R
    B = mean(Y)
        
    :param img: image to analyze, in RGB
    :type img: numpy.ndarray
    :return: mean brightness
    :rtype: float
    """
    # Ensure the image is in RGB format (using numpy arrays)
    img = np.array(img)
    
    # Vectorized calculation of brightness using the BT.601 formula
    B = np.mean(np.dot(img[..., :3], [0.299, 0.587, 0.114]))  # Dot product for each pixel
    
    return B  # Return the mean brightness value
