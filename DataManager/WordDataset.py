import os
import cv2
import csv  # Import CSV module to properly parse the CSV file
import numpy as np
from torch.utils import data
from Graphemes.utils import skip_chars
from ImageProcessing.image_processing import adjust_contrast_grey
import random

class WordDataset(data.Dataset):
    def __init__(self, img_dir, label_file_path, virtual_size=-1, inp_h=32, inp_w=128, transform=None):
        """
            label_file is a CSV file with the following format:
            filename1,word1
            filename2,word2
            ....
            filenameN,wordN
            All filenames should have the same length
            No header should be present
            Each file is an image of a word
        """
        self.inp_h = inp_h
        self.inp_w = inp_w
        self.transform = transform
        self.virtual_size = virtual_size

        # Use CSV reader to properly parse the file, handling quoted text properly
        with open(label_file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            img_paths = []
            words = []
            for row in reader:
                if len(row) == 2:  # Ensure the CSV row has two columns
                    img_path = os.path.join(img_dir, row[0].strip())  # Path to image with leading/trailing whitespace removed
                    word = row[1].strip().strip('"')  # Remove leading/trailing whitespace and double quotes
                    img_paths.append(img_path)
                    words.append(word)

        # Filter out words that contain characters from skip_chars
        filtered_words = []
        filtered_img_paths = []
        for i, word in enumerate(words):
            if any((c in skip_chars) for c in word):
                continue
            filtered_words.append(word)
            filtered_img_paths.append(img_paths[i])

        self.data = list(zip(filtered_img_paths, filtered_words))
        
    def __len__(self):
        if self.virtual_size == -1:
            return len(self.data)
        else:
            return self.virtual_size

    def __getitem__(self, idx):
        if self.virtual_size != -1:
            idx = random.randint(0, len(self.data)-1)
            
        image = cv2.imread(self.data[idx][0])
        if image is None:
            raise FileNotFoundError(f"Image not found: {self.data[idx][0]}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = adjust_contrast_grey(image)
        img_h, img_w = image.shape
        image = cv2.resize(image, (0, 0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)
        image = np.reshape(image, (self.inp_h, self.inp_w, 1))

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = image.transpose(2, 0, 1)

        return image, self.data[idx][1]
