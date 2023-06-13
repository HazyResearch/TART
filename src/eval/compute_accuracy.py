import sklearn.metrics as metrics
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from typing import List, Dict
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os
from typing import Tuple
import pickle


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accuracy computation")
    parser.add_argument(
        "--path_to_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="EleutherAI/gpt-neo-125m",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./accuracies",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="accuracies.pkl",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="hatespeech_18",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
    )
  
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    model_name_split = args.model_name.split("/")[-1]

    if os.path.exists(args.data_dir) is False:
        os.mkdir(args.data_dir)

    if args.method in ["TART", "base"]:
        path_to_save = f"{args.data_dir}/{args.dataset}/{model_name_split}/{args.method}/{args.run_id}/{args.checkpoint}/"
    else:
        path_to_save = (
            f"{args.data_dir}/{args.dataset}/{model_name_split}/{args.method}/"
        )
    if os.path.exists(path_to_save) is False:
        os.makedirs(path_to_save)

    data = pd.read_pickle(args.path_to_file)
    all_accuracies = compute_accuracy(data)

    # save all_accuracies as a pkl file
    pickle_name = os.path.join(path_to_save, args.file_name)

    with open(pickle_name, "wb") as f:
        pickle.dump(all_accuracies, f)

