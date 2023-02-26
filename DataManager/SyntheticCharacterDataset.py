from torch.utils.data import Dataset
import cv2
import os
import numpy as np
class SyntheticCharacterDataset(Dataset):

    inp_h = 32
    inp_w = 32

    @classmethod
    def load_character_image(cls, img_path, transpose=True):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_h, img_w = image.shape
        image = cv2.resize(image, (0,0), fx=cls.inp_w/img_w, fy=cls.inp_h/img_h, interpolation=cv2.INTER_CUBIC)
        image = np.reshape(image, (cls.inp_h, cls.inp_w, 1))
        if transpose:
            image = image.transpose(2, 0, 1)
        return image

    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]

        img_path = os.path.join(self.img_dir, img_name)
        label = int(img_name.split('_')[0])

        if self.transform is not None:
            image = SyntheticCharacterDataset.load_character_image(img_path, False)
            image = self.transform(image = image)["image"]
        else:
            image = SyntheticCharacterDataset.load_character_image(img_path)

        return image/255.0, label