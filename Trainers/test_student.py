import json

from Models.Student.Student import Student
from DataManager.WordLoader import get_word_loader

with open('Graphemes/Extracted/graphemes_bw_bnhtrd_syn.json', 'r') as f:
    graphemes_dict = json.load(f)

student  = Student(graphemes_dict)

test_imgs = ['/kaggle/working/curated_dataset/test/images']
test_labels = ['/kaggle/working/curated_dataset/test/labels.csv']

test_loader, test_size = get_word_loader(test_imgs, test_labels, augment=False)

model_path = '/kaggle/working/ML-Project-CSE-472/ML-Project-Files/SavedModels/teacher_ResNet18_135.pt'

student.load_model(model_path)
student.validate(1, test_loader, save_best=False)