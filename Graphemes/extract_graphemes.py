from .utils import normalize_word, ads_grapheme_extraction, skip_chars, merge_csv_files, get_graphemes_dict
import json
import torch
from tqdm import tqdm

def decode_prediction(pred, inv_graphemes_dict):
    grapheme_list = []
    grapheme_id_list = []

    for i in range(len(pred)):
        if pred[i] != 0 and (i == 0 or pred[i] != pred[i-1]):
            grapheme_list.append(inv_graphemes_dict[pred[i]])
            grapheme_id_list.append(pred[i])

    return grapheme_id_list, ''.join(grapheme_list)

def decode_label(label, inv_graphemes_dict):
    decoded = []
    for i in range(len(label)):
        if label[i] != 0:
            decoded.append(inv_graphemes_dict[label[i]])
    return ''.join(decoded)

def words_to_labels(words, graphemes_dict):
    labels = []
    lengths = []
    maxlen = 0
    for word in words:
        word = normalize_word(word)
        label = []
        for grapheme in ads_grapheme_extraction(word):
            label.append(graphemes_dict[grapheme])
        labels.append(label)
        lengths.append(len(label))
        maxlen = max(len(label), maxlen)

    # pad all labels to the same length - maxlen of current batch
    for i in range(len(labels)):
        labels[i] = labels[i] + [0]*(maxlen-len(labels[i]))
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    lengths = torch.tensor(lengths, dtype=torch.long).to(device)

    return labels, lengths

def extract_grapheme_labels(label_files_paths, graphemes_dict=None):
    """
        label_files_paths is a list of paths to csv files
        Each csv file should have the following format:
        filename1,word1
        filename2,word2
        ....
        filenameN,wordN
        All filenames should be have the same length
        No header should be present
        Each file is an image of a word
    """
    labels = []
    words_processed = []
    lengths = []

    lines = merge_csv_files(label_files_paths)
    filenames = [line.split(",")[0] for line in lines]
    words_raw = [line[len(line.split(",")[0])+1:] for line in lines]

    if graphemes_dict is None:
        graphemes_dict = get_graphemes_dict(words_raw)

    for word in words_raw:
        if any((c in skip_chars) for c in word):
            continue
        word = normalize_word(word)
        words_processed.append(word)

        try:
            label = []
            for grapheme in ads_grapheme_extraction(word):
                label.append(graphemes_dict[grapheme])
            labels.append(label)
            lengths.append(len(label))
        except KeyError:
            raise KeyError(f"Grapheme {'grapheme'} not found in graphemes_dict")

    inv_graphemes_dict = {v: k for k, v in graphemes_dict.items()}

    return {
        'graphemes_dict': graphemes_dict,
        'inv_grapheme_dict': inv_graphemes_dict,
        'words': words_processed,
        'labels': labels,
        'lengths': lengths,
        'filenames': filenames,
    }

if __name__ == '__main__':
    bw_train = f'Datasets/BanglaWriting/train/labels.csv'
    bw_val = f'Datasets/BanglaWriting/val/labels.csv'
    bnhtrd_train = f'Datasets/Bn-HTRd/train/labels.csv'
    bnhtrd_val = f'Datasets/Bn-HTRd/val/labels.csv'
    syn_words = f'Datasets/SyntheticWords/labels.csv'

    all = [bw_train, bw_val, bnhtrd_train, bnhtrd_val, syn_words]
    dataset_name = 'bw_bnhtrd_syn'

    res = extract_grapheme_labels(all)
    graphemes_dict = res['graphemes_dict']
    inv_grapheme_dict = res['inv_grapheme_dict']

    with open(f"graphemes_{dataset_name}.json", "w") as f:
        json.dump(graphemes_dict, f)

    with open(f"inv_graphemes_{dataset_name}.json", "w") as f:
        json.dump(inv_grapheme_dict, f)

    with open(f"graphemes_{dataset_name}.txt", "w") as f:
        for grapheme in graphemes_dict:
            f.write(grapheme + "---" + str(graphemes_dict[grapheme]) + "\n")

    with open(f"inv_graphemes_{dataset_name}.txt", "w") as f:
        for grapheme in inv_grapheme_dict:
            f.write(str(grapheme) + "---" + inv_grapheme_dict[grapheme] + "\n")
    

    