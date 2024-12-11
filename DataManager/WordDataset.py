import os
import cv2
import numpy as np
from torch.utils import data
from Graphemes.utils import skip_chars
from ImageProcessing.image_processing import adjust_contrast_grey
import random

class WordDataset(data.Dataset):
    def __init__(self, img_dir, label_file_path, virtual_size=-1, inp_h=32, inp_w=128, transform=None):
        """
            label_file is a csv file with the following format:
            filename1,word1
            filename2,word2
            ....
            filenameN,wordN
            All filenames should be have the same length
            No header should be present
            Each file is an image of a word
        """
        self.inp_h = inp_h
        self.inp_w = inp_w
        self.transform = transform
        self.virtual_size = virtual_size

        label_file = open(label_file_path, "r").readlines()
        img_names = [line.split(",")[0] for line in label_file]
        img_paths = [os.path.join(img_dir, img_name) for img_name in img_names]
        words = [line[len(img_names[0])+1:].strip() for line in label_file]

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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = adjust_contrast_grey(image)
        img_h, img_w = image.shape
        image = cv2.resize(image, (0,0), fx=self.inp_w/img_w, fy=self.inp_h/img_h, interpolation=cv2.INTER_CUBIC)
        image = np.reshape(image, (self.inp_h, self.inp_w, 1))

        if self.transform is not None:
            image = self.transform(image = image)["image"]
        else:
            image = image.transpose(2, 0, 1)
        
           
        return image, self.data[idx][1]