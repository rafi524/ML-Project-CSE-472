import json

from Models.Student.Student import Student
from DataManager.WordLoader import get_word_loader

with open('Graphemes/Extracted/graphemes_bw_bnhtrd_syn.json', 'r') as f:
    graphemes_dict = json.load(f)

student  = Student(graphemes_dict)

test_datasets = [
    {
        'img_dir': '/kaggle/input/ml-project-472/bangla-writing/bangla-writing/images',
        'label_file_path': '/kaggle/working/labels.csv',
    }
]

test_loader, test_size = get_word_loader(test_datasets, augment=False)
print(f"Test data loaded. Size: {test_size}")

model_path = '/kaggle/working/ML-Project-CSE-472/ML-Project-Files/Checkpoints/student_VGG_noteacher_basic'

student.load_model(model_path)
student.test(test_loader, save_best=False)