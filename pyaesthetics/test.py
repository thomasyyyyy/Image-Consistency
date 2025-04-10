import time
import numpy as np

# Sample image (e.g., 1000x1000 pixels with RGB channels)
img = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)

# Optimized function
def optimized_luminance(img):
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    brightness_op = np.mean(R) * 0.2126 + np.mean(G) * 0.7152 + np.mean(B) * 0.0722
    print(brightness_op)
    return brightness_op

# Original function with flattening and reshaping
def original_luminance(img):
    img = np.array(img).flatten()
    img = img.reshape(int(len(img) / 3), 3)
    img = np.transpose(img)
    brightness_or = np.mean(img[0]) * 0.2126 + np.mean(img[1]) * 0.7152 + np.mean(img[2]) * 0.0722
    print(brightness_or)
    return brightness_or

# Timing the optimized function
start = time.time()
optimized_luminance(img)
end = time.time()
print(f"Optimized function took {end - start} seconds.")


# Timing the original function
start = time.time()
original_luminance(img)
end = time.time()
print(f"Original function took {end - start} seconds.")

