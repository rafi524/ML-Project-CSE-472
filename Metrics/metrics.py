import pandas as pd
import Levenshtein
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np

def print_metric(name, value, round_digits=4):
    print(f"{name.rjust(40)}: {str(round(value, round_digits)).rjust(10)}")
    
def accuracy_metrics(y_true, y_pred, n_classes, target_names, final_action='print'):
    try:
        report = classification_report(y_true, y_pred, labels=np.arange(1, n_classes), 
                                    zero_division=0, output_dict=True, 
                                    target_names=target_names)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
    except:
        f1_micro = 0
        f1_macro = 0
        accuracy = 0

    if final_action == 'print' or final_action == 'both':
        print_metric('Accuracy', accuracy)
        print_metric('F1 Micro', f1_micro)
        print_metric('F1 Macro', f1_macro)

    if final_action == 'return' or final_action == 'both':
        return {
            'accuracy': accuracy,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
        }
    
def recognition_metrics(predictions, labels, final_action='print', res_file_path=None):
    all_num = correct_num = norm_edit_dis = total_edit_dist = total_length = 0
    
    if not res_file_path is None:
        res_dict = {'label':[], 'pred':[], 'edit_dist':[], 'label_len':[]}

    for pred, label in zip(predictions, labels):
        edit_dist = Levenshtein.distance(pred, label)
        max_len = max(len(pred), len(label), 1)
      
        norm_edit_dis += edit_dist / max_len
        
        total_edit_dist += edit_dist
        total_length += max_len
        
        if edit_dist == 0:
            correct_num += 1
        all_num += 1
        
        if not res_file_path is None:          
            res_dict['label'].append(label)
            res_dict['pred'].append(pred)
            res_dict['edit_dist'].append(edit_dist)
            res_dict['label_len'].append(len(label))
    
    if not res_file_path is None:
        res_df = pd.DataFrame(res_dict)
        res_df.to_csv(res_file_path, mode='w', index=False, header=True) 
        
    results = {
        'abs_match': correct_num,
        'wrr': correct_num / all_num,
        'total_ned': norm_edit_dis,
        'crr': 1 - total_edit_dist / total_length
    }
    
    if final_action.lower().strip() == 'print' or final_action.lower().strip() == 'both':
        print_metric('Absolute Word Match Count', results['abs_match'], 0)
        print_metric('Word Recognition Rate (WRR)', results['wrr'])
        print_metric('Normal Edit Distance (NED)', int(results['total_ned']), 0)
        print_metric('Character Recognition Rate (CRR)', results['crr'])

    if final_action.lower().strip() == 'return' or final_action.lower().strip() == 'both':
        return results