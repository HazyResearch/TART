from datasets import load_dataset
from sklearn.model_selection import train_test_split
import os
import pandas as pd

from abc import ABC, abstractmethod


def prep_train_split(train_df, total_train_samples, seed, k_range=None):
    # sample a class balanced set of train samples from train_df
    my_list = train_df["label"]

    train_df_samples = (
        train_df.groupby("label")
        .apply(lambda x: x.sample(int(total_train_samples / 2), random_state=seed))
        .reset_index(drop=True)
    )

    samples = []
    total_sampled = 0
    for k in k_range:
        curr_k = k - total_sampled
        total_sampled = k
        # sample k samples from each class of train_df_samples, and remove the sampled samples from train_df_samples
        k_samples = train_df_samples.groupby("label").apply(
            lambda x: x.sample(int(curr_k / 2), random_state=seed)
        )
        # get second level index
        k_samples = k_samples.droplevel(0)
        train_df_samples = train_df_samples.drop(k_samples.index)
        samples.append(k_samples)

    # concatenate all the samples as a final df
    final_df = pd.concat(samples)
    return final_df


def get_dataset(
    dataset_name,
    data_dir_path,
    save_data=True,
    cache_dir=None,
    total_train_samples=256,
    seed=42,
    k_range=None,
    input_key="text",
):
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)

    df = dataset["train"].to_pandas()
    df = df.rename(columns={"sms": "text"})

    if "test" in dataset.keys() and dataset_name != "sst2":
        train_df = dataset["train"].to_pandas()
        test_df = dataset["test"].to_pandas()
    else:
        train_df, test_df = train_test_split(df, test_size=0.75)
        print(f"Train size: {len(train_df)}")

    if save_data:
        if not os.path.exists(os.path.join(data_dir_path, dataset_name)):
            os.makedirs(os.path.join(data_dir_path, dataset_name))

        train_df.to_csv(
            os.path.join(data_dir_path, f"{dataset_name}/{dataset_name}_train.csv"),
            index=False,
        )
        test_df.to_csv(
            os.path.join(data_dir_path, f"{dataset_name}/{dataset_name}_test.csv"),
            index=False,
        )

    train_samples = prep_train_split(
        train_df, total_train_samples, seed, k_range=k_range
    )

    X_train = train_samples[input_key].tolist()
    y_train = train_samples["label"].tolist()
    X_test = test_df[input_key].tolist()
    y_test = test_df["label"].tolist()

    return X_train, y_train, X_test, y_test


def load_data_mm(
    data_path,
    dataset="sms",
    input_key="image",
    seed=42,
    pos_class=0,
    neg_class=1,
    max_train_samples=256,
):
    if dataset == "cifar10":
        dataset = load_dataset("cifar10")

    elif dataset == "mnist":
        input_key = "image"
        dataset = load_dataset("mnist")

    elif dataset == "speech_commands":
        dataset = load_dataset("speech_commands", "v0.01")

    else:
        dataset = load_dataset(dataset)

    dataset_train = (
        dataset["train"]
        .filter(lambda example: example["label"] in [pos_class, neg_class])
        .map(lambda example: {"label": 0 if example["label"] == pos_class else 1})
    )
    dataset_test = (
        dataset["test"]
        .filter(lambda example: example["label"] in [pos_class, neg_class])
        .map(lambda example: {"label": 0 if example["label"] == pos_class else 1})
    )

    dataset_train_1 = (
        dataset_train.filter(lambda example: example["label"] == 1)
        .shuffle(seed=seed)
        .select(range(int(max_train_samples / 2)))
    )
    # get k samples from dataset_test where label = 0
    dataset_train_0 = (
        dataset_train.filter(lambda example: example["label"] == 0)
        .shuffle(seed=seed)
        .select(range(int(max_train_samples / 2)))
    )

    return (
        dataset_train_1[input_key],
        dataset_train_1["label"],
        dataset_train_0[input_key],
        dataset_train_0["label"],
    ), (dataset_test[input_key], dataset_test["label"])
