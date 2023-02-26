import numpy as np

def contrast_grey(img):
    high = np.percentile(img, 90)
    low  = np.percentile(img, 10)
    if high == low:
        high = np.percentile(img, 95)
        low  = np.percentile(img, 5)
    return (high-low)/(high+low), high, low

def adjust_contrast_grey(img, target = 0.4):
    contrast, high, low = contrast_grey(img)
    if high == low:
        return img
    if contrast < target:
        img = img.astype(int)
        ratio = 200./(high-low)
        img = (img - low + 25)*ratio
        img = np.maximum(np.full(img.shape, 0) ,np.minimum(np.full(img.shape, 255), img)).astype(np.uint8)
    return img