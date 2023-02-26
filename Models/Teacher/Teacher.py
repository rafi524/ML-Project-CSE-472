import torch
from .BasicConv import BasicConv
from .ResNet18 import ResNet18
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from tqdm import tqdm
import os
from Metrics.metrics import print_metric
from DataManager.SyntheticCharacterDataset import SyntheticCharacterDataset
import torch.nn.functional as F
import random
from tqdm import tqdm

class Teacher():

    def __init__(self, teacher_type, n_classes):
        self.teacher_type = teacher_type
        self.n_classes = n_classes
        self.prediction_dict = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if teacher_type == 'BasicConv':
            self.model =  BasicConv(n_classes)
        elif teacher_type == 'ResNet18':
            self.model = ResNet18(n_classes)
        else:
            raise ValueError('Teacher type not supported')

        self.model.to(self.device)

    def train(self, train_loader, val_loader, save_dir, n_epochs=10, lr=0.001, verbose_freq=1):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        start_time = time.time()
        self.metrics = {'freq': verbose_freq, 'train_loss': [], 'train_acc': [], 
                        'val_loss': [], 'val_acc': [], 'best_val_acc': 0, 'best_epoch': 0}
        self.save_dir = save_dir

        self.model.train()
        for epoch in range(1, n_epochs+1):
            running_loss = 0.0
            print(f"Epoch {epoch}/{n_epochs}:")
            for images, labels in tqdm(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            if epoch % verbose_freq == 0:
                self.print_metrics(train_loader, val_loader, epoch, start_time, running_loss)

        print("Training finished.")

    def print_metrics(self, train_loader, val_loader, epoch, start_time, running_loss):
        epoch_loss = running_loss / len(train_loader)
        train_acc = self.get_accuracy(train_loader)
        val_acc = self.get_accuracy(val_loader)

        print("_"*75)
        print_metric('Training Loss', 100 * epoch_loss, 2)
        print_metric('Training accuracy', 100 * train_acc, 2)
        print_metric('Validation accuracy', 100 * val_acc, 2)
        print_metric('Time elapsed (seconds)', round(time.time() - start_time), 0)
        print("_"*75)

        self.metrics['train_loss'].append(epoch_loss)
        self.metrics['val_loss'].append(epoch_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['val_acc'].append(val_acc)

        if val_acc > self.metrics['best_val_acc']:
            prev = self.save_dir + f"/teacher_{self.teacher_type}_{str(self.metrics['best_epoch']).zfill(3)}.pt"
            if os.path.exists(prev):
                os.remove(prev)
            self.metrics['best_val_acc'] = val_acc
            self.metrics['best_epoch'] = epoch
            self.save_model(self.save_dir + f"/teacher_{self.teacher_type}_{str(self.metrics['best_epoch']).zfill(3)}.pt")

    def get_accuracy(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return accuracy

    def predict_img(self, img_path, t=1, target=-1):
        self.model.eval()
        img = SyntheticCharacterDataset.load_character_image(img_path)
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2]) / 255.0
        img = torch.from_numpy(img).float().to(self.device)
        
        with torch.no_grad():
            logits = self.model(img)
        probs = F.softmax(logits/t, dim=1) if target != 0 else F.log_softmax(logits/t, dim=1)
        pred = probs.data.cpu().argmax().item()

        return pred, probs

    def generate_prediction_dict(self, saved_path, img_dir, t):
        self.load_model(saved_path)
        self.prediction_dict = {}
        all_imgs = os.listdir(img_dir)
        
        for class_no in range(1, self.n_classes):
            class_str = str(class_no).zfill(3)
            class_imgs = [img for img in all_imgs if img.startswith(class_str)]

            while True:
                rand_img = class_imgs[random.randint(0, len(class_imgs)-1)]
                img_path = os.path.join(img_dir, rand_img)
                pred, probs = self.predict_img(img_path, t=t)

                if pred == class_no:
                    break

            self.prediction_dict[class_str]= probs

    def get_stacked_probs(self, labels):
        assert self.prediction_dict is not None, "Teacher's Prediction dictionary is not initialized"
        
        batch_probs = torch.zeros((len(labels), 31, self.n_classes))
        for label_no, label in enumerate(labels):
            label = list(filter(lambda grapheme_id: grapheme_id != 0, label))
            label = [grapheme_id.item() for grapheme_id in label]
            
            probs = torch.zeros((31, self.n_classes))
            probs_no = 0
            for grapheme_no, grapheme_id in enumerate(label, 1):
                for _ in range(31//len(label)):
                    probs[probs_no, :] = self.prediction_dict[str(grapheme_id).zfill(3)]
                    probs_no += 1

                if len(label) == grapheme_no:
                    while probs_no <= 30:
                        probs[probs_no, :] = self.prediction_dict[str(grapheme_id).zfill(3)]
                        probs_no += 1

            batch_probs[label_no, :, :] = probs
        
        batch_probs = batch_probs.transpose(0,1)
        return batch_probs

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))