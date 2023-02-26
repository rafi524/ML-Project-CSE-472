import numpy as np
import random
from PIL import Image

def crop(img, bg_color, height=32, width=32, min_margin=0, max_margin=20):
    """Crops around the text. Margin around the text is random."""
    img = np.array(img)

    first_row = np.where(img.sum(axis=1) < bg_color * img.shape[1])[0][0]
    last_row = np.where(img.sum(axis=1) < bg_color * img.shape[1])[0][-1]
    first_col = np.where(img.sum(axis=0) < bg_color * img.shape[0])[0][0]
    last_col = np.where(img.sum(axis=0) < bg_color * img.shape[0])[0][-1]

    first_row = max(0, first_row-random.randint(min_margin, 20))
    first_col = max(0, first_col-random.randint(min_margin, 20))
    last_row = min(img.shape[0], last_row+random.randint(min_margin, max_margin))
    last_col = min(img.shape[1], last_col+random.randint(min_margin, max_margin))

    img = img[first_row:last_row, first_col:last_col]
    img = Image.fromarray(img)
    img = img.resize((width, height))

    return img