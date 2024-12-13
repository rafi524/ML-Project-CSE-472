import torch
import torch.optim as optim
from tqdm import tqdm
from torch import nn
import numpy as np
import os
import shutil
import warnings
import json
import pandas as pd

from Graphemes.extract_graphemes import decode_prediction, decode_label, words_to_labels
from .CRNN import CRNN
from Metrics.metrics import recognition_metrics, accuracy_metrics, print_metric
from Models.Teacher.Teacher import Teacher

class Student:

    def __init__(slf, graphemes_dict, extractor_type='VGG', use_attention=False, teacher_data=None, variant=""):

        slf.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        slf.graphemes_dict = graphemes_dict
        slf.inv_graphemes_dict = {v: k for k, v in graphemes_dict.items()}
        slf.n_classes = len(graphemes_dict)+1
        slf.teacher_data = teacher_data

        slf.student_type = 'student_' + extractor_type
        slf.student_type += f"_teacher_{teacher_data['teacher_type']}" if teacher_data is not None else '_noteacher'
        slf.student_type += '_'+variant if len(variant) > 0 else ''
        
        slf.model = CRNN(extractor_type, use_attention, slf.n_classes)
        slf.model.to(slf.device)
        slf.ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True).to(slf.device)

    def init_teacher(slf):
        slf.teacher = Teacher(slf.teacher_data['teacher_type'], n_classes=slf.n_classes)
        slf.teacher.generate_prediction_dict(slf.teacher_data['saved_path'], slf.teacher_data['img_dir'], t=slf.t)

    def init_training(slf, lr):
        slf.optimizer = optim.Adam(filter(lambda p: p.requires_grad, slf.model.parameters()), lr, weight_decay=1e-05)
        slf.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(slf.optimizer, T_0=15, T_mult=1, eta_min=0.0001)
        slf.alpha = 0.5
        slf.t = 2
        slf.metrics = {
                       'Train': {'loss': [], 'wrr': [], 'crr': [], 'accuracy': [], 'total_ned': [], 'abs_match': [], 'f1_micro': [], 'f1_macro': []}, 
                       'Validation': {'loss': [], 'wrr': [], 'crr': [], 'accuracy': [], 'total_ned': [], 'abs_match': [], 'f1_micro': [], 'f1_macro': []},
                       'Epochs': {'total': 0, 'best': 0, 'latest': 0},
                       'Best': {'wrr': 0}
                      }

    def init_epoch(slf, epoch, train=True):
        if train:
            slf.model.train()
        else:
            slf.model.eval()
        slf.y_true = []
        slf.y_pred = []
        slf.decoded_preds = []
        slf.decoded_labels = []
        slf.batch_loss = 0
        slf.epoch = epoch
        if slf.teacher_data is not None:
            slf.init_teacher()

    def add_teacher_loss(slf, loss, logits, labels):
        probs = torch.nn.functional.log_softmax(logits/slf.t , dim=2)
        teacher_probs = slf.teacher.get_stacked_probs(labels).cuda()

        # UserWarning: reduction: 'mean' divides the total loss by both the batch size and the 
        # support size.'batchmean' divides only by the batch size, and aligns with the KL div 
        # math definition.'mean' will be changed to behave the same as 'batchmean' in the next 
        # major release. 
        warnings.filterwarnings('ignore', category=UserWarning)
        ty = nn.KLDivLoss(reduction='mean')(probs , teacher_probs)
        warnings.filterwarnings('default', category=UserWarning)
        
        loss = ty * (slf.t*slf.t * 2.0 + slf.alpha) + loss * (1.-slf.alpha)
        return loss
    
    def forward(slf, images, words=None):
        images = images.to(slf.device).float() / 255.
        logits = slf.model(images)
        probs = torch.nn.functional.log_softmax(logits , dim=2)
        slf.batch_size = images.size(0)

        if words is not None:
            labels, label_lengths = words_to_labels(words, slf.graphemes_dict)
            probs_size = torch.tensor([probs.size(0)] * slf.batch_size, dtype=torch.long).to(slf.device)

            loss = slf.ctc_loss(probs, labels, probs_size, label_lengths)
            if slf.teacher_data is not None:
                loss = slf.add_teacher_loss(loss, logits, labels)
            slf.batch_loss += loss.item()  

            return probs, labels, loss
        
        return probs

    def init_checkpointing(slf, epochs, resume, checkpoint_root):
        slf.checkpoint_folder = os.path.join(checkpoint_root, slf.student_type)

        slf.model_ckt = os.path.join(slf.checkpoint_folder, 'model.pt')
        slf.optimizer_ckt = os.path.join(slf.checkpoint_folder, 'optimizer.pt')
        slf.scheduler_ckt = os.path.join(slf.checkpoint_folder, 'scheduler.pt')
        slf.metrics_ckt = os.path.join(slf.checkpoint_folder, 'metrics.json')

        slf.best_model = os.path.join(slf.checkpoint_folder, 'best_model.pt')
        slf.best_samples = os.path.join(slf.checkpoint_folder, 'best_samples.txt')

        if resume:
            assert os.path.exists(slf.checkpoint_folder), f"Check Point Folder Not Found: {slf.checkpoint_folder}"
            slf.load_checkpoint()
            slf.total_epochs = slf.metrics['Epochs']['total']
            slf.starting_epoch = slf.metrics['Epochs']['latest'] + 1
        else:
            slf.total_epochs = epochs
            slf.starting_epoch = 1
            slf.metrics['Epochs']['total'] = epochs
            shutil.rmtree(slf.checkpoint_folder) if os.path.exists(slf.checkpoint_folder) else None
            os.mkdir(slf.checkpoint_folder)

    def train(slf, train_loader, val_loader, checkpoint_root, resume=False, epochs=30, lr = 0.0003):
        slf.init_training(lr)

        slf.init_checkpointing(epochs, resume, checkpoint_root)
        for epoch in range(slf.starting_epoch, slf.total_epochs+1):
            slf.init_epoch(epoch)
            print("Training Epoch: ", str(epoch)+"/"+str(slf.total_epochs))
            for images, words in tqdm(train_loader):
                probs, labels, loss = slf.forward(images, words)
                              
                slf.optimizer.zero_grad()
                loss.backward()
                slf.optimizer.step()

                slf.save_mini_batch_results(probs, labels)
        
            slf.scheduler.step()
            slf.print_stats('Train', save_best=False)
            slf.validate(epoch, val_loader)

            slf.metrics['Epochs']['latest'] = epoch
            slf.save_checkpoint()

            print("="*125)
    
    def validate(slf, epoch, val_loader, save_best=True):
        slf.init_epoch(epoch, train=False)
        with torch.no_grad():
            print("Validating:")
            for images, words in tqdm(val_loader):
                probs, labels, loss = slf.forward(images, words)
                slf.save_mini_batch_results(probs, labels)
            slf.print_stats('Validation', save_best=save_best)

    def save_mini_batch_results(slf, probs, labels):
        _, preds = probs.max(2)
        preds = preds.transpose(1, 0).contiguous().detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        for pred, label in zip(preds, labels):
            grapheme_id_list, grapheme = decode_prediction(pred, slf.inv_graphemes_dict)

            slf.decoded_preds.append(grapheme)
            slf.decoded_labels.append(decode_label(label, slf.inv_graphemes_dict))

            min_len = min(len(grapheme_id_list), len(label))
            slf.y_true.extend(grapheme_id_list[:min_len])
            slf.y_pred.extend(list(label)[:min_len])

    def append_metrics(slf, data_set, wrr, crr, total_ned, abs_match, accuracy, f1_micro, f1_macro):
        slf.metrics[data_set]['loss'].append(slf.batch_loss/slf.batch_size)
        slf.metrics[data_set]['wrr'].append(wrr)
        slf.metrics[data_set]['crr'].append(crr)
        slf.metrics[data_set]['total_ned'].append(total_ned)
        slf.metrics[data_set]['abs_match'].append(abs_match)
        slf.metrics[data_set]['accuracy'].append(accuracy)
        slf.metrics[data_set]['f1_micro'].append(f1_micro)
        slf.metrics[data_set]['f1_macro'].append(f1_macro)
        
    def print_stats(slf, data_set, save_best):
        print_metric(f"{data_set} loss", slf.batch_loss/slf.batch_size)
        results = recognition_metrics(slf.decoded_preds, slf.decoded_labels, final_action='both')
        wrr, crr, total_ned, abs_match = results['wrr'], results['crr'], results['total_ned'], results['abs_match']

        results = accuracy_metrics(slf.y_true, slf.y_pred, slf.n_classes, final_action='both',
                         target_names=[v for _, v in slf.inv_graphemes_dict.items()])
        accuracy, f1_micro, f1_macro = results['accuracy'], results['f1_micro'], results['f1_macro']
        
        slf.append_metrics(data_set, wrr, crr, total_ned, abs_match, accuracy, f1_micro, f1_macro)
        
        slf.print_samples()

        if save_best and wrr > slf.metrics['Best']['wrr']:
            slf.metrics['Best']['wrr'] = wrr
            slf.metrics['Epochs']['best'] = slf.epoch
            slf.save_best_model()

    def save_best_model(slf):
        os.remove(slf.best_model) if os.path.exists(slf.best_model) else None 
        os.remove(slf.best_samples) if os.path.exists(slf.best_samples) else None
        
        torch.save(slf.model.state_dict(), slf.best_model)
        with open(slf.best_samples, 'w') as f:
            for i in range(len(slf.decoded_preds)):
                f.write(f"{slf.decoded_labels[i]}::{slf.decoded_preds[i]}\n")

    def print_samples(slf, sample_size=5):
        total = len(slf.decoded_preds)
        sample = np.random.choice(total, sample_size, replace=False)
        print("Actual :: Predicted", end="  |||  ")
        for i in sample:
            print(f"{slf.decoded_labels[i]} :: {slf.decoded_preds[i]}", end="  |||  ")
        print("\n")

    def load_checkpoint(slf):
        slf.model.load_state_dict(torch.load(slf.model_ckt))
        slf.optimizer.load_state_dict(torch.load(slf.optimizer_ckt))
        slf.scheduler.load_state_dict(torch.load(slf.scheduler_ckt))
        slf.metrics = json.load(open(slf.metrics_ckt, 'r'))

    def save_checkpoint(slf):
        torch.save(slf.model.state_dict(), slf.model_ckt)
        torch.save(slf.optimizer.state_dict(), slf.optimizer_ckt)
        torch.save(slf.scheduler.state_dict(), slf.scheduler_ckt)
        json.dump(slf.metrics, open(slf.metrics_ckt, 'w'), indent=4)

#  def load_checkpoint(slf):
#         slf.model.load_state_dict(torch.load(slf.model_ckt))
#         slf.optimizer.load_state_dict(torch.load(slf.optimizer_ckt))
#         slf.scheduler.load_state_dict(torch.load(slf.scheduler_ckt))
#         slf.metrics = json.load(open(slf.metrics_ckt, 'r'))

    def load_model(self, model_path):
        """
        Load a pre-trained model from the specified path.
        """
        model_ckt = os.path.join(model_path, 'best_model.pt')
        optimizer_ckt = os.path.join(model_path, 'optimizer.pt')
        scheduler_ckt = os.path.join(model_path, 'scheduler.pt')
        metrics_ckt = os.path.join(model_path, 'metrics.json')

        self.model.load_state_dict(torch.load(model_ckt))
        # self.optimizer.load_state_dict(torch.load(optimizer_ckt))
        # self.scheduler.load_state_dict(torch.load(scheduler_ckt))
        self.metrics = json.load(open(metrics_ckt, 'r'))
        self.model.eval()

    #  def validate(slf, epoch, val_loader, save_best=True):
    #     slf.init_epoch(epoch, train=False)
    #     with torch.no_grad():
    #         print("Validating:")
    #         for images, words in tqdm(val_loader):
    #             probs, labels, loss = slf.forward(images, words)
    #             slf.save_mini_batch_results(probs, labels)
    #         slf.print_stats('Validation', save_best=save_best)
    def test(slf, test_loader, save_best=False, save_path='test_results.csv'):
        """
        Test the model and save predictions and labels into a CSV file.
        
        Args:
            slf: The model instance containing the testing logic.
            test_loader: The data loader for test data.
            save_best: Whether to save the best results.
            save_path: File path to save the predictions and labels.
        """
        slf.init_epoch(1, train=False)
        results = {'true_words': [], 'pred_words': []}
        
        with torch.no_grad():
            print("Testing:")
            for images, words in tqdm(test_loader):
                probs, labels, loss = slf.forward(images, words)
                slf.save_mini_batch_results(probs, labels)
            
            # Collect decoded predictions and labels for saving
            results['true_words'] = slf.decoded_labels
            results['pred_words'] = slf.decoded_preds

            # Save results to a CSV file
            results_df = pd.DataFrame(results)
            results_df.to_csv(save_path, index=False)

            slf.print_samples()
            slf.print_stats('Test', save_best=save_best)
            print(f"Test results saved to {save_path}")




                

