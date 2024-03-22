import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix

from samsiamese import SiameseNetwork
from datasets import EvaluateImagePairDataset, EvaluatePairDataset
from eval_metrics import evaluate, plot_roc
from utils import init_evaluate_log, write_csv


def eval_bce(checkpoint_path, test_pairs, tensors):
    out_path = './eval_results/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # model = "vgg128"
    # csv = f"{out_path}/problematiccoins.csv"
    # init_evaluate_log(csv)

    model = SiameseNetwork()
    model.to('cuda')

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    print("epoch: ", checkpoint['epoch'])

    # test_dataset = EvaluateImagePairDataset(test_pairs, tensors, transform='resnet50')
    test_dataset = EvaluatePairDataset(test_pairs, tensors)
    test_loader = DataLoader(test_dataset, shuffle=False)

    # Lists to store predictions and labels
    all_preds, all_labels, false_positives, false_negatives = [], [], [], []


    model.eval()
    with torch.no_grad():
        for tensor1, tensor2, label, coin1, coin2 in test_loader:
            tensor1, tensor2, label = map(lambda x: x.to('cuda'), [tensor1, tensor2, label])
            label = label.view(-1)
            
            prob = model(tensor1, tensor2).squeeze(1)

            predictions = (prob > 0.5).cpu().numpy()
            labels = label.cpu().numpy()
            # Store predictions and labels
            all_preds.extend(predictions)
            all_labels.extend(labels)

            # Store coin names for false positives and negatives
            for i, (pred, true_label) in enumerate(zip(predictions, labels)):
                if pred == 1 and true_label == 0:
                    false_positives.append((coin1[i], coin2[i]))
                elif pred == 0 and true_label == 1:
                    false_negatives.append((coin1[i], coin2[i]))

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    cm = confusion_matrix(all_labels, all_preds)

    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    false_pr = float(fp/(fp+tn))
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)

    # write_csv(csv, [accuracy, precision, recall, f1, auc, false_pr, tpr, fpr])

    # Print metrics
    print(f'TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}')
    print(f'Accuracy: {accuracy:.3f}')
    print(f'Accuracy (from cm): {(tp+tn)/(tp+tn+fp+fn):.3f}')
    print(f'Precision: {precision:.3f}')
    print(f'Recall: {recall:.3f}')
    print(f'FPR: {false_pr:.4f}')
    print(f'F1 Score: {f1:.3f}')
    print(f'AUC: {auc:.3f}')

    # Export false positives and false negatives to CSV
    false_positives_df = pd.DataFrame(false_positives, columns=['Coin1', 'Coin2'])
    false_negatives_df = pd.DataFrame(false_negatives, columns=['Coin1', 'Coin2'])

    # Assign 'Result' column after creating the DataFrames
    error_df = pd.concat([
        false_positives_df.assign(Result='False Positive'), 
        false_negatives_df.assign(Result='False Negative')
    ], ignore_index=True)

    error_csv = f"{out_path}/error_analysis.csv"
    error_df.to_csv(error_csv, index=False)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    eval_bce(
    './best.pt',
    'data/ccc_tensors/pair/combined/test_dataset.csv',
    'data/ccc_tensors/data/'
)
