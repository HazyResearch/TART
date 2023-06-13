import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def load_model(model_name, path_to_finetuned_model=None, cache_dir=None):
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        pad_token="<|pad|>",
        cache_dir=cache_dir,
    )
    tokenizer.truncation_side = "left"

    # should work for all generative LM models
    trained_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        config=config,
        cache_dir=cache_dir,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        config=config,
        cache_dir=cache_dir,    
    )

    # load weights
    if path_to_finetuned_model is not None:
        synthetic_model = torch.load(path_to_finetuned_model)

        new_state_dict = {}
        if "pythia" in model_name:
            for k, v in synthetic_model.items():
                if k.startswith("_backbone.gpt_neox"):
                    k = k.replace("_backbone.gpt_neox", "gpt_neox")
                    new_state_dict[k] = v
                elif k.startswith("_backbone.embed_out"):
                    k = k.replace("_backbone.embed_out", "embed_out")
                    new_state_dict[k] = v
                elif k.startswith("_backbone.embed_in"):
                    k = k.replace("_backbone.embed_in", "embed_in")
                    new_state_dict[k] = v
                else:
                    new_state_dict[k] = v
        elif "opt" in model_name:
            for k, v in synthetic_model.items():
                if k.startswith("_backbone.model"):
                    k = k.replace("_backbone.model", "model")
                    new_state_dict[k] = v
                elif k.startswith("_backbone.lm_head"):
                    k = k.replace("_backbone.lm_head", "lm_head")
                    new_state_dict[k] = v
                else:
                    new_state_dict[k] = v
        else:
            for k, v in synthetic_model.items():
                if k.startswith("_backbone.transformer"):
                    k = k.replace("_backbone.transformer", "transformer")
                    new_state_dict[k] = v
                elif k.startswith("_backbone.lm_head"):
                    k = k.replace("_backbone.lm_head", "lm_head")
                    new_state_dict[k] = v
                else:
                    new_state_dict[k] = v

        # load state dict
        trained_model.load_state_dict(new_state_dict, strict=False)

    trained_model.eval()
    base_model.eval()

    return (
        trained_model.cuda(),
        base_model.cuda(),
        tokenizer,
    )


# Data load function
def load_data(data_path, dataset="sms", input_key="sms", seed=42):
    # load training data
    train_set_path = os.path.join(data_path, f"{dataset}/train_samples_s{seed}.csv")
    training_set = pd.read_csv(train_set_path)
    X_train = training_set[input_key].tolist()
    y_train = training_set["label"].tolist()

    # load test data
    if "ag_news" in dataset or "dbpedia" in dataset or "civil_comments" in dataset:
        test_set_path = os.path.join(data_path, f"{dataset}/test_samples_bal.csv")
    else:
        test_set_path = os.path.join(data_path, f"{dataset}/test_samples_orig.csv")
    test_set = pd.read_csv(test_set_path)
    X_test = test_set[input_key].tolist()
    y_test = test_set["label"].tolist()

    # return
    return (X_train, y_train), (X_test, y_test)


def generate_in_context_example(
    X_train, y_train, template, seed, order=None, text_threshold=100
):
    samples = list(zip(X_train, y_train))
    # generate a random permutation of 0 to 128
    if not order:
        # set numpy random seed
        np.random.seed(seed)
        order = np.random.permutation(len(samples))

    in_context_example = ""
    ys = []
    for i in order:
        sample = samples[int(i)]
        if sample[1] == 1:
            label = "positive"
            ys.append(sample[1])
        else:
            label = "negative"
            ys.append(sample[1])
        in_context_example += template.format(
            sentence=sample[0][0:text_threshold].strip(), label=label
        )

    return in_context_example, ys


def get_template(template):
    if template == "sentence_label":
        train_template = f"Sentence: {{sentence:}}\nLabel: {{label:}}\n"
        test_template = f"Sentence: {{sentence:}}\nLabel:"
    elif template == "colon_label":
        train_template = f"{{sentence:}} : {{label:}} , "
        test_template = f"{{sentence:}} :"
    return train_template, test_template


def load_data_mm(
    data_path, dataset="mnist", input_key="image", seed=42, pos_class=0, neg_class=1, max_train_samples=256
):
    if dataset == "cifar10":
        dataset = load_dataset("cifar10")

    elif dataset == "mnist":
        input_key = "image"
        dataset = load_dataset("mnist")

    elif dataset == "speech_commands":
        dataset = load_dataset("speech_commands", "v0.01")

    dataset_train = (
        dataset["train"]
        .filter(lambda example: example["label"] in [pos_class, neg_class])
        .map(lambda example: {"label": 0 if example["label"] == neg_class else 1})
    )
    dataset_test = (
        dataset["test"]
        .filter(lambda example: example["label"] in [pos_class, neg_class])
        .map(lambda example: {"label": 0 if example["label"] == neg_class else 1})
    )

    dataset_train_1 = (
        dataset_train.filter(lambda example: example["label"] == 1)
        .shuffle(seed=seed)
        .select(range(int(max_train_samples/2)))
    )
    # get k samples from dataset_test where label = 0
    dataset_train_0 = (
        dataset_train.filter(lambda example: example["label"] == 0)
        .shuffle(seed=seed)
        .select(range(int(max_train_samples/2)))
    )

    return (
        dataset_train_1[input_key],
        dataset_train_1["label"],
        dataset_train_0[input_key],
        dataset_train_0["label"],
    ), (dataset_test[input_key], dataset_test["label"])
