import os
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import auc, confusion_matrix


def evaluate_bce(preds, labels):
    preds = np.array(preds).flatten()
    labels = np.array(labels).flatten()

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / float(tp + tn + fp + fn)
    return tpr, fpr, acc

'''
Used for finding optimal Threshold value for classifying similar / dissimilar pairs
on Siamese Network trained with Contrastive and Triplet Loss
'''
def evaluate(distances, labels, step=0.001):
    min_threshold = min(distances)
    max_threshold = max(distances)
    max_acc = 0
    max_tpr = 0
    max_fpr = 0
    best_threshold = min_threshold
    issame = (labels == 1)

    for threshold in np.arange(min_threshold, max_threshold+step, step):
        tp = (distances <= threshold) & issame
        tn = (distances > threshold) & (~issame)
        tnr = tn.sum().astype(float) / (~issame).sum().astype(float) 
        fp = (distances <= threshold) & (~issame)
        fn = (distances > threshold) & issame

        tpr = tp.sum().astype(float) / issame.sum().astype(float)  
        fpr = fp.sum().astype(float) / (~issame).sum().astype(float)  

        acc = (tp.sum() + tn.sum()).astype(float) / (tp.sum() + tn.sum() + fp.sum() + fn.sum()).astype(float)

        # max_acc = max(acc, max_acc)
        # max_tpr = max(tpr, max_tpr)
        # max_fpr = max(fpr, max_fpr)
        if acc > max_acc:
            max_acc = acc
            best_threshold = threshold
            max_tpr = tpr
            max_fpr = fpr

    return max_tpr, max_fpr, max_acc, threshold

def final_evaluate(distances, labels, threshold):
    issame = (labels == 1)

    tp = (distances <= threshold) & issame
    tn = (distances > threshold) & (~issame)
    fp = (distances <= threshold) & (~issame)
    fn = (distances > threshold) & issame

    acc = (tp.sum() + tn.sum()).astype(float) / (tp.sum() + tn.sum() + fp.sum() + fn.sum()).astype(float)
    recall = tp.sum().astype(float) / issame.sum().astype(float)
    fpr = fp.sum().astype(float) / (~issame).sum().astype(float)  
    precision = tp.sum().astype(float) / (tp.sum() + fp.sum()).astype(float)

    # Calculate F1 score
    if precision + recall == 0:  # Avoid division by zero
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return acc, precision, recall, fpr, f1_score

def plot_roc(fpr, tpr, out_path, figure_name="roc.png"):
    plt.switch_backend('Agg')

    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='#16a085',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='#2c3e50', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right", frameon=False)
    fig.savefig(os.path.join(out_path, figure_name, dpi=fig.dpi))

def plot_loss(train_losses, val_losses, out_path):
    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses, label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(out_path, 'loss_plot.jpg'))
    plt.close()

def plot_accuracy(train_accs, val_accs, out_path):
    plt.plot(train_accs, label='train acc')
    plt.plot(val_accs, label='val acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(out_path, 'accuracy_plot.jpg'))
    plt.close()