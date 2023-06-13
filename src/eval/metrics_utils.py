import sklearn.metrics as metrics
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from typing import List, Dict

import numpy as np



def compute_accuracy(data: Dict) -> torch.Tensor:
    epoch_keys = sorted(list(data[list(data.keys())[0]].keys()))
    seed_keys = sorted(list(data.keys()))
    k_keys = sorted(list(data[list(data.keys())[0]][epoch_keys[0]].keys()))
    all_accuracies = torch.zeros(len(epoch_keys), len(k_keys), len(seed_keys))
    for i_epoch, epoch in enumerate(epoch_keys):
        for i_seed, seed in enumerate(seed_keys):
            for i_key, k in enumerate(k_keys):
                predict_labels = data[seed][epoch][k]["predicted_label"]
                gt_labels = data[seed][epoch][k]["gt_label"]
                comp = [x == y for x, y in zip(predict_labels, gt_labels)]
                accuracy = sum(comp) / len(comp)
                all_accuracies[i_epoch, i_key, i_seed] = accuracy

    return all_accuracies


def compute_confusion_matrix(data: Dict) -> List[torch.Tensor]:
    epoch_keys = list(data[list(data.keys())[0]].keys())
    seed_keys = list(data.keys())
    k_keys = list(data[list(data.keys())[0]][epoch_keys[0]].keys())
    true_positive = torch.zeros(len(epoch_keys), len(k_keys), len(seed_keys))
    true_negative = torch.zeros(len(epoch_keys), len(k_keys), len(seed_keys))
    false_positive = torch.zeros(len(epoch_keys), len(k_keys), len(seed_keys))
    false_negative = torch.zeros(len(epoch_keys), len(k_keys), len(seed_keys))
    accuracies = torch.zeros(len(epoch_keys), len(k_keys), len(seed_keys))

    for i_epoch, epoch in enumerate(epoch_keys):
        for i_seed, seed in enumerate(seed_keys):
            for i_key, k in enumerate(data[seed][epoch].keys()):
                predict_labels = data[seed][epoch][k]["predicted_label"]
                predict_scores = data[seed][epoch][k]["predicted_scores"]
                predict_labels_per_score = []
                for x in predict_scores:
                    if x[0] > x[2]:
                        predict_labels_per_score.append("positive")
                    else:
                        predict_labels_per_score.append("negative")

                gt_labels = data[seed][epoch][k]["gt_label"]
                tn, fp, fn, tp = confusion_matrix(
                    gt_labels, predict_labels_per_score, labels=["negative", "positive"]
                ).ravel()
                true_positive[i_epoch, i_key, i_seed] = tp / (tp + fn)
                true_negative[i_epoch, i_key, i_seed] = tn / (tn + fp)
                false_positive[i_epoch, i_key, i_seed] = fp / (fp + tn)
                false_negative[i_epoch, i_key, i_seed] = fn / (fn + tp)
                accuracies[i_epoch, i_key, i_seed] = (tp + tn) / (tp + tn + fp + fn)

    return true_positive, true_negative, false_positive, false_negative, accuracies


def compute_confusion_matrix_label(data: Dict) -> List[torch.Tensor]:
    epoch_keys = list(data[list(data.keys())[0]].keys())
    seed_keys = list(data.keys())
    k_keys = list(data[list(data.keys())[0]][epoch_keys[0]].keys())
    true_positive = torch.zeros(len(epoch_keys), len(k_keys), len(seed_keys))
    true_negative = torch.zeros(len(epoch_keys), len(k_keys), len(seed_keys))
    false_positive = torch.zeros(len(epoch_keys), len(k_keys), len(seed_keys))
    false_negative = torch.zeros(len(epoch_keys), len(k_keys), len(seed_keys))
    null_positive = torch.zeros(len(epoch_keys), len(k_keys), len(seed_keys))
    null_negative = torch.zeros(len(epoch_keys), len(k_keys), len(seed_keys))

    for i_epoch, epoch in enumerate(epoch_keys):
        for i_seed, seed in enumerate(seed_keys):
            for i_key, k in enumerate(data[seed][epoch].keys()):
                predict_labels = data[seed][epoch][k]["predicted_label"]
                predict_labels_three_class = []
                for x in predict_labels:
                    if x.strip() not in ["positive", "negative"]:
                        predict_labels_three_class.append("null")
                    else:
                        predict_labels_three_class.append(x.strip())

                gt_labels = data[seed][epoch][k]["gt_label"]
                tn, fp, en, fn, tp, ep, _, _, _ = confusion_matrix(
                    gt_labels,
                    predict_labels_three_class,
                    labels=["negative", "positive", "null"],
                ).ravel()
                num_pos = tp + fn + ep
                num_neg = tn + fp + en

                true_positive[i_epoch, i_key, i_seed] = tp / num_pos
                true_negative[i_epoch, i_key, i_seed] = tn / num_neg
                null_positive[i_epoch, i_key, i_seed] = ep / num_pos
                false_positive[i_epoch, i_key, i_seed] = fp / num_neg
                false_negative[i_epoch, i_key, i_seed] = fn / num_pos
                null_negative[i_epoch, i_key, i_seed] = en / num_neg

    return (
        true_positive,
        true_negative,
        false_positive,
        false_negative,
        null_positive,
        null_negative,
    )


def ensemble_accuracy(datasets: List[Dict]) -> torch.Tensor:
    d = datasets[0]
    epoch_keys = list(d[list(d.keys())[0]].keys())
    seed_keys = list(d.keys())
    k_keys = list(d[list(d.keys())[0]][epoch_keys[0]].keys())
    all_accuracies = torch.zeros(len(epoch_keys), len(k_keys), len(seed_keys))

    label_map = {"positive": 1, "negative": 0}
    label_map_inverse = {1: "positive", 0: "negative", 2: "null"}

    for i_epoch, epoch in enumerate(epoch_keys):
        for i_seed, seed in enumerate(seed_keys):
            for i_key, k in enumerate(d[seed][epoch].keys()):
                preds = []
                for data in datasets:
                    new_labels = []
                    for pl in data[seed][epoch][k]["predicted_label"]:
                        if pl.strip() not in label_map.keys():
                            new_labels.append(2)
                        else:
                            new_labels.append(label_map[pl.strip()])
                    preds.append(np.array(new_labels))

                majority_vote = np.apply_along_axis(
                    lambda x: np.bincount(x).argmax(), axis=0, arr=np.array(preds)
                )
                predict_labels = [label_map_inverse[x] for x in majority_vote]
                gt_labels = d[seed][epoch][k]["gt_label"]
                comp = [x == y for x, y in zip(predict_labels, gt_labels)]
                accuracy = sum(comp) / len(comp)
                all_accuracies[i_epoch, i_key, i_seed] = accuracy

    return all_accuracies


