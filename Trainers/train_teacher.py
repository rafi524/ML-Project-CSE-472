from DataManager.SyntheticCharacterLoader import SyntheticCharacterLoader
from Models.Teacher.Teacher import Teacher
import json

with open('Graphemes/Extracted/graphemes_bw_bnhtrd_syn.json', 'r') as f:
    graphemes_dict = json.load(f)

n_classes = len(graphemes_dict)+1
teacher = Teacher('ResNet18', n_classes=n_classes)

train_loader = SyntheticCharacterLoader('Datasets/SyntheticCharacters/train',  batch_size=1024)
val_loader = SyntheticCharacterLoader('Datasets/SyntheticCharacters/val', batch_size=1024)

save_path = 'ML-Project-Files/SavedModels'
teacher.train(train_loader, val_loader, save_path, n_epochs=5, lr=0.001, verbose_freq=5)