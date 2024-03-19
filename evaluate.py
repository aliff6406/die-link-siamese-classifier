import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix

import tripletsiamese
import siamese
from offline_pair import OfflinePairDataset
from eval_metrics import evaluate, plot_roc

import config

def evaluate():
    test_pairs = config.obverse_test
    obverse_tensors = config.obverse_tensors

    model = siamese.SiameseNetwork()
    model.to('cuda')

    checkpoint_path = 'runs/bce-fix-val/best.pt'
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    print("epoch: ", checkpoint['epoch'])

    test_dataset = OfflinePairDataset(pair_dir=test_pairs, tensor_dir=obverse_tensors)
    test_loader = DataLoader(test_dataset, shuffle=False)

    # Lists to store predictions and labels
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for tensor1, tensor2, label, tensor1_name, tensor2_name in test_loader:
            tensor1, tensor2, label = map(lambda x: x.to('cuda'), [tensor1, tensor2, label])
            label = label.view(-1)

            prob = model(tensor1, tensor2).squeeze(1)

            # Store predictions and labels
            all_preds.extend((prob > 0.5).cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    cm = confusion_matrix(all_labels, all_preds)

    tn, fp, fn, tp = cm([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    
    # Calculate metrics
    accuracy = (all_labels == (all_preds > 0.5)).mean()
    precision = precision_score(all_labels, all_preds > 0.5)
    recall = recall_score(all_labels, all_preds > 0.5)
    f1 = f1_score(all_labels, all_preds > 0.5)
    auc = roc_auc_score(all_labels, all_preds)

    # Print metrics
    print(f'TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}')
    print(f'Accuracy: {accuracy}')
    print(f'Accuracy (from cm): {(tp+tn)/(tp+tn+fp+fn)}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'AUC: {auc}')

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()


def eval_contrastive():
    
    pass

def eval_triplet():
    test_dir = config.obverse_test_dir
    test_csv = config.obverse_test_csv

    model = tripletsiamese.SiameseNetwork()
    model.to('cuda')

    checkpoint_path = 'runs/triplet-sgd/best.pt'
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])

    test_dataset = SiameseTensorPairDataset(label_dir=test_csv, tensor_dir=test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    distances = []
    labels = []

    model.eval()
    with torch.no_grad():
        for (tensor1, tensor2), label in test_loader:
            tensor1, tensor2 = tensor1.to('cuda'), tensor2.to('cuda')

            emb1 = model.forward_once(tensor1)
            emb2 = model.forward_once(tensor2)

            # Calculate euclidean distance between image embeddings
            distance = F.pairwise_distance(emb1, emb2)
            distance = distance.squeeze().cpu().numpy()
            label = label.squeeze().cpu().numpy()
            distances.append(distance)
            labels.append(label)

            print(distance)
            print(label)

    distances = np.array(distances)
    labels = np.array(labels)

    tpr, fpr, accuracy = evaluate(distances, labels)

    print("accuracy: ", np.mean(accuracy))
    print("tpr: ", np.mean(tpr))
    print("fpr: ", np.mean(fpr))

if __name__ == "__main__":
    eval_bce()
    # eval_bce()
