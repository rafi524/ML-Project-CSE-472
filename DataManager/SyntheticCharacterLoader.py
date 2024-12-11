import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from .SyntheticCharacterDataset import SyntheticCharacterDataset

data_transform = A.Compose([  
        A.Rotate(limit=30, p=0.5),
        A.Blur(blur_limit=3, p=0.25),
        A.OpticalDistortion(p=0.25),
        A.GridDistortion(p=0.25),
        A.ElasticTransform(alpha=0.5, sigma=1, alpha_affine=0, p=0.25),
        A.GaussNoise(var_limit=(120.0, 135.0), mean=0, always_apply=False, p=0.25),
        ToTensorV2(),
    ])
class SyntheticCharacterLoader(DataLoader):
    def __init__(self, data_dir, batch_size=256, num_workers=1, shuffle=True):
        self.dataset = SyntheticCharacterDataset(data_dir, transform=data_transform)
        super().__init__(self.dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)