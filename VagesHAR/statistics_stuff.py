from __future__ import division, print_function

import itertools
import json
import os

import numpy as np

import matplotlib
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

matplotlib.use("Agg")  # Choose non-interactive background
import matplotlib.pyplot as plt


def save_confusion_matrix_image(matrix, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.summer,
                                save_path="./whatever.png"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    row_sums = matrix.sum(axis=1)
    shade_matrix = (matrix.transpose() / row_sums).transpose()

    plt.clf()
    plt.imshow(shade_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.tick_params(labelsize=7)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, matrix[i, j],
                 horizontalalignment="center",
                 color="black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)


def generate_and_save_statistics_json(y_true, y_pred, number_to_class_name_dict, save_path):
    precision, recall, f_score, support = precision_recall_fscore_support(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    print(os.path.split(save_path)[1], accuracy)

    occurring_classes = sorted(list(set(y_true)))

    classes = [number_to_class_name_dict[c] for c in occurring_classes]

    save_statistics_json(accuracy, classes, f_score, precision, recall, save_path, support)


def save_statistics_json(accuracy, classes, f_score, precision, recall, save_path, support):
    recall_dict = dict([(c, s) for c, s in zip(classes, recall)])
    precision_dict = dict([(c, s) for c, s in zip(classes, precision)])
    f_score_dict = dict([(c, s) for c, s in zip(classes, f_score)])
    support_dict = dict([(c, s) for c, s in zip(classes, support)])
    d = {
        "recall": recall_dict,
        "precision": precision_dict,
        "f_score": f_score_dict,
        "support": support_dict,
        "accuracy": accuracy
    }
    with open(save_path, "w") as f:
        json.dump(d, f)


def generate_and_save_confusion_matrix(y_true, y_pred, number_to_label_dict, save_path, title=""):
    matrix = confusion_matrix(y_true, y_pred)
    original_labels = set(y_true)
    all_occurring_classes = set(y_pred) | original_labels
    class_names = [number_to_label_dict[x] for x in sorted(list(all_occurring_classes))]
    save_confusion_matrix_image(matrix, class_names, save_path=save_path, title=title)
