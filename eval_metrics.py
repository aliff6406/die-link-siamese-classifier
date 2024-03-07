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
Code adapted from David Sandberg's FaceNet implementation [https://github.com/davidsandberg/facenet/blob/master/src/lfw.py]
'''

def evaluate(distances, labels, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 5, 0.01)
    tpr, fpr, accuracy, fp, fn = calculate_roc(thresholds, distances, labels, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy


def calculate_roc(thresholds, distances, labels, nrof_folds=10):
    nrof_pairs = min(len(labels), len(distances))
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    indices = np.arange(nrof_pairs)    

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the optimal threshold for the current fold
        acc_train = np.zeros((nrof_thresholds))

        # Calculate the accuracy at each threshold on the train set
        for threshold_idx, threshold in enumerate(thresholds):
            # Calulate accuracy at each threshold value
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, 
                                                                distances[train_set], 
                                                                labels[train_set])
        # Best threshold value based on highest accuracy
        best_threshold_index = np.argmax(acc_train)
        # Calculate the tprs and fprs 
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold, 
                                                                                                 distances[test_set], 
                                                                                                 labels[test_set])
        # Use the best threshold to get the accuracy on the test set
        _, _, accuracy[fold_idx] = calculate_accuracy(threshold[best_threshold_index],
                                                      distances[test_set],
                                                      labels[test_set])
        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


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