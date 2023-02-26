import json

from Models.Student.Student import Student
from DataManager.WordLoader import get_word_loader

with open('Graphemes/Extracted/graphemes_bw_bnhtrd_syn.json', 'r') as f:
    graphemes_dict = json.load(f)

student  = Student(graphemes_dict)

test_imgs = ['Datasets/ILM/train', 'Datasets/ILM/val']
test_labels = ['Datasets/ILM/train/labels.csv', 'Datasets/ILM/val/labels.csv']

test_loader, test_size = get_word_loader(test_imgs, test_labels, augment=False)

model_path = '/content/drive/MyDrive/ML-Project-Files/SavedModels/student_VGG_hasteacher_ResNet18_045.pt'

student.load_model(model_path)
student.validate(1, test_loader, save_best=False)