import json

from Models.Student.Student import Student
from DataManager.WordLoader import get_word_loader

print("Loading graphemes dictionary...")
with open('Graphemes/Extracted/graphemes_bw_bnhtrd_syn.json', 'r') as f:
    graphemes_dict = json.load(f)
print("Graphemes dictionary loaded.")

teacher_data = {
    'teacher_type': 'ResNet18',
    'saved_path': 'ML-Project-Files/SavedModels/teacher_ResNet18_085.pt',
    'img_dir': 'Datasets/SyntheticCharacters/train'
}

teacher_data = None
variant = 'basic'
epochs = 150
use_attention = False

print("Initializing Student model...")
student  = Student(graphemes_dict, teacher_data=teacher_data, use_attention=use_attention, variant=variant)
print("Student model initialized.")

train_datasets = [
        {
            'img_dir': '/kaggle/input/ml-project-472/compress/merge_100',
            'label_file_path': '/kaggle/input/ml-project-472/compress/change_100.csv',
        }
    ]

val_datasets = [
        {
            'img_dir': '/kaggle/input/ml-project-472/compress/merge_1',
            'label_file_path': '/kaggle/input/ml-project-472/compress/change_1.csv'
        }
       
    ]

print("Loading training data...")
train_loader, train_size = get_word_loader(train_datasets, augment=True)
print(f"Training data loaded. Size: {train_size}")

print("Loading validation data...")
val_loader, val_size = get_word_loader(val_datasets, augment=False)
print(f"Validation data loaded. Size: {val_size}")

checkpoint_root='ML-Project-Files/Checkpoints'

print("Starting training...")
student.train(train_loader, val_loader, checkpoint_root=checkpoint_root, resume=False, epochs=epochs)
print("Training completed.")