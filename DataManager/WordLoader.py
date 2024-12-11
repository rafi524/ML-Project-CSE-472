import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import ConcatDataset, DataLoader
from .WordDataset import WordDataset

data_transform_aug = A.Compose([ 
        A.Rotate(limit=10, p=0.5),
        A.Blur(blur_limit=3, p=0.25),
        A.OpticalDistortion(p=0.25),
        A.GridDistortion(p=0.25),
        A.ElasticTransform(alpha=0.5, sigma=1, alpha_affine=0, p=0.25),
        A.GaussNoise(var_limit=(120.0, 135.0), mean=0, always_apply=False, p=0.25),       
        ToTensorV2(),
    ])

data_transform_aug_synthetic = A.Compose([ 
        A.Rotate(limit=10, p=0.5),
        A.Blur(blur_limit=3, p=0.5),
        A.OpticalDistortion(p=0.5),
        A.GridDistortion(p=0.5),
        A.ElasticTransform(alpha=0.5, sigma=1, alpha_affine=0, p=0.5),
        A.GaussNoise(var_limit=(120.0, 135.0), mean=0, always_apply=False, p=0.5),       
        ToTensorV2(),
    ])

data_transform_no_aug = A.Compose([ 
        ToTensorV2(),
    ])

def get_word_loader(datasets, augment=True, batch_size=256, num_workers=1, shuffle=True):
    dataset_objects = []
    for dataset in datasets:
        img_dir, label_file_path = dataset['img_dir'], dataset['label_file_path']
        virtual_size = -1 if 'virtual_size' not in dataset else dataset['virtual_size']

        if augment:
            if 'synthetic' in dataset and dataset['synthetic']:
                transform = data_transform_aug_synthetic
            else:
                transform = data_transform_aug
        else:
            transform = data_transform_no_aug

        dataset_objects.append(WordDataset(img_dir, label_file_path, transform=transform, virtual_size=virtual_size))

    merged_dataset = ConcatDataset(dataset_objects)
    word_loader = DataLoader(merged_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return word_loader, len(word_loader)