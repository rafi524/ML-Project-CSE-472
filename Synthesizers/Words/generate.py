from PIL import Image, ImageDraw, ImageFont
import os
import shutil 

import random
from tqdm import tqdm
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
dst = 'Datasets/SyntheticWords/all'
if os.path.exists(dst):
    shutil.rmtree(dst)
os.mkdir(dst)

fonts_root = 'Synthesizers/fonts'
font_files = os.listdir(fonts_root)
font_files.sort()

lines = open('Datasets/SyntheticWords/labels.csv', 'r', encoding='utf-8').readlines()
lines = [line.strip() for line in lines]
words = [line[len(line.split(',')[0])+1:] for line in lines]

def generate_word_image(word, font_file, height, width):
    b_color = random.randint(175, 255)
    t_color = random.randint(0, 75)

    image = Image.new("L", (300, 100), (b_color))
    font_size = random.randint(20, 40)
    font = ImageFont.truetype(os.path.join(fonts_root, font_file), font_size)
    draw = ImageDraw.Draw(image)
    draw.text((30, 20), word, font=font, fill=(t_color))
    image = crop(image, b_color, height=height, width=width, min_margin=10, max_margin=30)

    return image

def word_generator(height, width):
    word = random.choice(words)
    font_file = random.choice(font_files)
    return generate_word_image(word, font_file, height, width), word

if __name__ == "__main__":
    labels = ""
    for word_no, word in enumerate(tqdm(words), 1):
        for font_no, font_file in enumerate(font_files, 1):
            filename = str(word_no).zfill(5) + '_' + str(font_no).zfill(2) + '.png'
            labels += filename + ',' + word + '\n'
            image = generate_word_image(word, font_file, 32, 128)
            image.save(os.path.join(dst, filename))

open(os.path.join(dst, 'labels.csv'), 'w', encoding='utf-8').write(labels)