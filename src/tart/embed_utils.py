import random

import numpy as np
import torch

import sys
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

sys.path.append("../")

from reasoning_module.models import TransformerModel


sigmoid = torch.nn.Sigmoid()

from transformers.utils import logging

logging.set_verbosity(40)

import torchvision.transforms as transforms
from tokenizers import Tokenizer
from typing import List, Dict


def get_embeds_stream_audio(
    tokenizer: Tokenizer,
    model: torch.nn.Module,
    X_test: List,
    x_k: List,
    y_k: List,
    k: int,
    seed: int = 42,
):
    random.seed(seed)

    feature_extractor = tokenizer
    embed_list_tr = []

    ids = list(range(k))
    random.shuffle(ids)

    x_k = [x_k[i] for i in ids]
    y_k = [y_k[i] for i in ids]

    for i, rec in enumerate(tqdm(x_k)):
        inputs = feature_extractor(rec["array"], return_tensors="pt")
        input_features = inputs.input_features
        decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id

        with torch.no_grad():
            outputs = model(
                input_features.cuda(), decoder_input_ids=decoder_input_ids.cuda()
            )

        last_hidden_states = outputs.encoder_last_hidden_state
        mean_last_hidden_states = torch.mean(last_hidden_states, dim=1)
        embed_list_tr.append(mean_last_hidden_states)

    embed_final_tr = torch.stack(embed_list_tr, axis=0).squeeze(1)

    embed_list_tst = []
    for rec in X_test:
        inputs = feature_extractor(rec["array"], return_tensors="pt")
        input_features = inputs.input_features
        decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id

        with torch.no_grad():
            outputs = model(
                input_features.cuda(), decoder_input_ids=decoder_input_ids.cuda()
            )

        last_hidden_states = outputs.encoder_last_hidden_state
        mean_last_hidden_states = torch.mean(last_hidden_states, dim=1)
        embed_list_tst.append(mean_last_hidden_states)

    embed_final_tst = torch.stack(embed_list_tst, axis=0).squeeze(1)

    return embed_final_tr, embed_final_tst, y_k


def get_embeds_stream_image(
    tokenizer: Tokenizer,
    model: torch.nn.Module,
    X_test: List,
    x_k: List,
    y_k: List,
    k: int,
    seed: int = 42,
):
    random.seed(seed)
    transform = transforms.Compose(
        [
            transforms.Resize((model.config.image_size, model.config.image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]
    )
    embed_list_tr = []

    ids = list(range(k))
    random.shuffle(ids)

    x_k = [x_k[i] for i in ids]
    y_k = [y_k[i] for i in ids]

    for image in tqdm(x_k):
        image = image
        image = transform(image)
        image = image.unsqueeze(0)

        with torch.no_grad():
            outputs = model(image.cuda())

        last_hidden_states = outputs.last_hidden_state
        mean_last_hidden_states = torch.mean(last_hidden_states, dim=1)
        embed_list_tr.append(mean_last_hidden_states)

    embed_final_tr = torch.stack(embed_list_tr, axis=0).squeeze(1)

    embed_list_tst = []
    for image in tqdm(X_test):
        image = transform(image)
        image = image.unsqueeze(0)

        with torch.no_grad():
            outputs = model(image.cuda())

        last_hidden_states = outputs.last_hidden_state
        mean_last_hidden_states = torch.mean(last_hidden_states, dim=1)
        embed_list_tst.append(mean_last_hidden_states)

    embed_final_tst = torch.stack(embed_list_tst, axis=0).squeeze(1)

    return embed_final_tr, embed_final_tst, y_k


def get_embeds_naive_repeat(
    tokenizer: Tokenizer,
    model: torch.nn.Module,
    X_test: List,
    x_k: List,
    y_k: List,
    k: int,
    thresh: int = 100,
    seed: int = 42,
):
    # tokenize
    label_dict = {0: "negative", 1: "positive"}
    random.seed(seed)
    ids = list(range(k))
    random.shuffle(ids)
    # Create the prompt

    # positive and negative token ids
    pos_token_id_space = tokenizer(" positive", truncation=False).input_ids[0]
    neg_token_id_space = tokenizer(" negative", truncation=False).input_ids[0]

    # list to store empty embedding
    embed_tr = []

    y_tr_shuffle = []
    prompt_base = ""
    for num_i in range(k):
        id_sel = ids[num_i]
        prompt_base += f"Sentence: {x_k[id_sel].strip()[0:thresh]}\nLabel: {label_dict[y_k[id_sel]]}\n"
        y_tr_shuffle.append(y_k[id_sel])

    for d_id in range(k):
        # Create the prompt, datapoint d_id is skipped, the others remain in place
        # Append the example d_id
        id_sel = ids[d_id]
        prompt = (
            prompt_base
            + f"Sentence: {x_k[id_sel].strip()[0:thresh]}\nLabel: {label_dict[y_k[id_sel]]}\n"
        )

        # tokenize the prompt
        encodings_dict = tokenizer(prompt, truncation=True, return_tensors="pt")
        input_ids = encodings_dict.input_ids

        ys_b = torch.where(
            (input_ids == pos_token_id_space) | (input_ids == neg_token_id_space),
            input_ids,
            torch.tensor(-100),
        )
        idxs = torch.where(ys_b != -100)[1]

        # Embed the input ids
        with torch.no_grad():
            embed = model(input_ids.cuda(), return_dict=True, output_hidden_states=True)

        idxs_np = idxs.numpy()
        hidden_id = -1

        embed_comp = embed.hidden_states[hidden_id].squeeze()
        embed_dp = torch.mean(embed_comp[idxs_np[-2] + 1 : idxs_np[-1], :], axis=0)
        embed_tr.append(embed_dp)

    embed_np_tr_cor = [tensor.detach().cpu().numpy() for tensor in embed_tr]
    X_tr_embed_cor_seq = np.stack(embed_np_tr_cor)

    # # Obtain test embeddings for each test point
    embed_tst = []
    with torch.no_grad():
        for i in tqdm(range(len(X_test))):
            # if i % 50 == 0:
            #    print(f"iter: {i}")
            prompt_tst = prompt_base + f"Sentence: {X_test[i].strip()[0:thresh]}"
            encodings_dict = tokenizer(prompt_tst, truncation=True, return_tensors="pt")
            input_ids = encodings_dict.input_ids
            ys_b_test = torch.where(
                (input_ids == pos_token_id_space) | (input_ids == neg_token_id_space),
                input_ids,
                torch.tensor(-100),
            )
            idxs_test = torch.where(ys_b_test != -100)[1]
            idxs_np_test = idxs_test.numpy()
            embed_dp_tst = model(
                input_ids.cuda(), return_dict=True, output_hidden_states=True
            )
            embed_comp = embed_dp_tst.hidden_states[hidden_id].squeeze()
            embed_dp = torch.mean(embed_comp[idxs_np_test[-1] + 1 :, :], axis=0)
            embed_tst.append(embed_dp.detach().cpu())

    embed_np_tst_cor = [tensor.numpy() for tensor in embed_tst]
    X_tst_embed_cor_seq = np.stack(embed_np_tst_cor)

    return X_tr_embed_cor_seq, X_tst_embed_cor_seq, y_tr_shuffle


def get_embeds_long_context(
    tokenizer: Tokenizer,
    model: torch.nn.Module,
    X_test: List,
    X_train: List,
    y_train: List,
    k: int,
    thresh: int = 100,
    seed: int = 42,
):
    import random

    random.seed(seed)
    # tokenize
    x_k = X_train[0:k]
    y_k = y_train[0:k]

    # randomly sample 64 examples from the training set with equal number of positive and negative examples

    label_dict = {0: "negative", 1: "positive"}
    random.seed(seed)
    ids = list(range(k))
    random.shuffle(ids)
    # Create the prompt

    # positive and negative token ids
    pos_token_id_space = tokenizer(" positive", truncation=False).input_ids[0]
    neg_token_id_space = tokenizer(" negative", truncation=False).input_ids[0]

    # list to store empty embedding
    embed_tr = []

    y_tr_shuffle = []
    prompt_base = ""

    # Sample 64 examples to serve as context for each of the 256 embeddings
    # develop base_prompt using these

    for d_id in range(k):
        # Create the prompt, datapoint d_id is skipped, the others remain in place
        prompt = ""  # "Classify the following sentences as positive or negative:\n"
        prompt_base = ""

        # Append the example d_id
        id_sel = ids[d_id]
        icl_ids = list(range(len(X_train)))
        icl_ids.remove(id_sel)

        icl_ids_0 = [i for i in icl_ids if y_train[i] == 0]
        icl_ids_1 = [i for i in icl_ids if y_train[i] == 1]
        icl_ids_0 = random.sample(icl_ids_0, 32)
        icl_ids_1 = random.sample(icl_ids_1, 32)
        icl_ids = icl_ids_0 + icl_ids_1
        random.shuffle(icl_ids)

        for id_ice in icl_ids:
            prompt_base += f"Sentence: {X_train[id_ice].strip()[0:thresh]}\nLabel: {label_dict[y_train[id_ice]]}\n"

        prompt = (
            prompt_base
            + f"Sentence: {x_k[id_sel].strip()[0:thresh]}\nLabel: {label_dict[y_k[id_sel]]}\n"
        )

        # tokenize the prompt
        encodings_dict = tokenizer(
            prompt, truncation=True, max_length=2048, return_tensors="pt"
        )
        input_ids = encodings_dict.input_ids

        ys_b = torch.where(
            (input_ids == pos_token_id_space) | (input_ids == neg_token_id_space),
            input_ids,
            torch.tensor(-100),
        )
        idxs = torch.where(ys_b != -100)[1]

        # Embed the input ids
        with torch.no_grad():
            embed = model(input_ids.cuda(), return_dict=True, output_hidden_states=True)

        idxs_np = idxs.numpy()
        hidden_id = -1
        embed_comp = embed.hidden_states[hidden_id].squeeze()
        embed_dp = torch.mean(embed_comp[idxs_np[-2] + 1 : idxs_np[-1], :], axis=0)
        embed_tr.append(embed_dp)
        y_tr_shuffle.append(y_k[id_sel])

    embed_np_tr_cor = [tensor.detach().cpu().numpy() for tensor in embed_tr]
    X_tr_embed_cor_seq = np.stack(embed_np_tr_cor)

    # # Obtain test embeddings for each test point
    embed_tst = []
    with torch.no_grad():
        for i in tqdm(range(len(X_test))):
            prompt_tst = prompt_base + f"Sentence: {X_test[i].strip()[0:thresh]}"
            encodings_dict = tokenizer(
                prompt_tst, max_length=2048, truncation=True, return_tensors="pt"
            )
            input_ids = encodings_dict.input_ids
            ys_b_test = torch.where(
                (input_ids == pos_token_id_space) | (input_ids == neg_token_id_space),
                input_ids,
                torch.tensor(-100),
            )
            idxs_test = torch.where(ys_b_test != -100)[1]
            idxs_np_test = idxs_test.numpy()
            embed_dp_tst = model(
                input_ids.cuda(), return_dict=True, output_hidden_states=True
            )
            embed_comp = embed_dp_tst.hidden_states[hidden_id].squeeze()
            embed_dp = torch.mean(embed_comp[idxs_np_test[-1] + 1 :, :], axis=0)
            embed_tst.append(embed_dp.detach().cpu())

    embed_np_tst_cor = [tensor.numpy() for tensor in embed_tst]
    X_tst_embed_cor_seq = np.stack(embed_np_tst_cor)

    return X_tr_embed_cor_seq, X_tst_embed_cor_seq, y_tr_shuffle


def get_embeds_loo(
    tokenizer: Tokenizer,
    model: torch.nn.Module,
    X_test: List,
    x_k: List,
    y_k: List,
    k: int,
    thresh: int = 100,
    seed: int = 42,
):
    # tokenize
    label_dict = {0: "negative", 1: "positive"}
    random.seed(seed)
    ids = list(range(k))
    random.shuffle(ids)
    # Create the prompt

    # positive and negative token ids
    pos_token_id_space = tokenizer(" positive", truncation=False).input_ids[0]
    neg_token_id_space = tokenizer(" negative", truncation=False).input_ids[0]

    # list to store empty embedding
    embed_tr = []

    for d_id in range(k):
        # Create the prompt, datapoint d_id is skipped, the others remain in place
        prompt = ""  # "Classify the following sentences as positive or negative:\n"
        y_tr_shuffle = []
        pos_d_id = 0

        for num_i in range(k):
            if num_i == d_id:
                pos_d_id = num_i
                continue
            else:
                id_sel = ids[num_i]
                prompt += f"Sentence: {x_k[id_sel].strip()[0:thresh]}\nLabel: {label_dict[y_k[id_sel]]}\n"
                y_tr_shuffle.append(y_k[id_sel])

        # Append the example d_id
        id_sel = ids[pos_d_id]
        prompt += f"Sentence: {x_k[id_sel].strip()[0:thresh]}\nLabel: {label_dict[y_k[id_sel]]}\n"
        y_tr_shuffle.append(y_k[id_sel])

        # tokenize the prompt
        encodings_dict = tokenizer(
            prompt, truncation=True, max_length=2048, return_tensors="pt"
        )
        input_ids = encodings_dict.input_ids

        ys_b = torch.where(
            (input_ids == pos_token_id_space) | (input_ids == neg_token_id_space),
            input_ids,
            torch.tensor(-100),
        )
        idxs = torch.where(ys_b != -100)[1]

        # Embed the input ids
        with torch.no_grad():
            embed = model(input_ids.cuda(), return_dict=True, output_hidden_states=True)

        idxs_np = idxs.numpy()
        hidden_id = -1
        embed_comp = embed.hidden_states[hidden_id].squeeze()
        embed_dp = torch.mean(embed_comp[idxs_np[-2] + 1 : idxs_np[-1], :], axis=0)
        embed_tr.append(embed_dp)

    embed_np_tr_cor = [tensor.detach().cpu().numpy() for tensor in embed_tr]
    X_tr_embed_cor_seq = np.stack(embed_np_tr_cor)

    # # Obtain test embeddings for each test point
    embed_tst = []
    with torch.no_grad():
        for i in tqdm(range(len(X_test))):
            prompt_tst = prompt + f"Sentence: {X_test[i].strip()[0:thresh]}"
            encodings_dict = tokenizer(
                prompt_tst, max_length=2048, truncation=True, return_tensors="pt"
            )
            input_ids = encodings_dict.input_ids
            ys_b_test = torch.where(
                (input_ids == pos_token_id_space) | (input_ids == neg_token_id_space),
                input_ids,
                torch.tensor(-100),
            )
            idxs_test = torch.where(ys_b_test != -100)[1]
            idxs_np_test = idxs_test.numpy()
            embed_dp_tst = model(
                input_ids.cuda(), return_dict=True, output_hidden_states=True
            )
            embed_comp = embed_dp_tst.hidden_states[hidden_id].squeeze()
            embed_dp = torch.mean(embed_comp[idxs_np_test[-1] + 1 :, :], axis=0)
            embed_tst.append(embed_dp.detach().cpu())

    embed_np_tst_cor = [tensor.numpy() for tensor in embed_tst]
    X_tst_embed_cor_seq = np.stack(embed_np_tst_cor)

    return X_tr_embed_cor_seq, X_tst_embed_cor_seq, y_tr_shuffle


def get_embeds_vanilla(
    tokenizer: Tokenizer,
    model: torch.nn.Module,
    X_test: List,
    x_k: List,
    y_k: List,
    k: int,
    thresh: int = 100,
    seed: int = 42,
):
    random.seed(seed)

    # tokenize
    label_dict = {0: "negative", 1: "positive"}
    ids = list(range(k))
    random.shuffle(ids)
    # Create the prompt
    prompt = ""
    y_tr_shuffle = []

    for num_i in range(k):
        id_sel = ids[num_i]
        prompt += f"Sentence: {x_k[id_sel].strip()[0:thresh]}\nLabel: {label_dict[y_k[id_sel]]}\n"
        y_tr_shuffle.append(y_k[id_sel])

    if "t5" in model.name_or_path:
        encodings_dict = tokenizer(prompt, truncation=True, return_tensors="pt")

    else:
        encodings_dict = tokenizer(
            prompt, truncation=True, max_length=2048, return_tensors="pt"
        )
    input_ids = encodings_dict.input_ids

    # Find the location of the labels (which is where a datapoint ends)
    pos_token_id_space = tokenizer(" positive", truncation=False).input_ids[0]
    neg_token_id_space = tokenizer(" negative", truncation=False).input_ids[0]
    ys_b = torch.where(
        (input_ids == pos_token_id_space) | (input_ids == neg_token_id_space),
        input_ids,
        torch.tensor(-100),
    )
    idxs = torch.where(ys_b != -100)[1]

    # get train embeds
    with torch.no_grad():
        embed = model(
            input_ids=input_ids.cuda(), return_dict=True, output_hidden_states=True
        )

    idxs_np = idxs.numpy()
    hidden_id = -1

    embed_comp = embed.hidden_states[hidden_id].squeeze()
    embed_tr = []

    assert (
        len(idxs_np) == k
    ), "Reduce the number of in-context examples or decrease text_threshold"

    for i in range(k):
        if i == 0:
            embed_dp = torch.mean(embed_comp[0 : idxs_np[i], :], axis=0)
        else:
            embed_dp = torch.mean(
                embed_comp[idxs_np[i - 1] + 1 : idxs_np[i], :], axis=0
            )
        embed_tr.append(embed_dp)

    embed_np_tr_cor = [tensor.detach().cpu().numpy() for tensor in embed_tr]
    X_tr_embed_cor = np.stack(embed_np_tr_cor)

    # # Obtain test embeddings for each test point
    embed_tst = []
    with torch.no_grad():
        for i in tqdm(range(len(X_test))):
            prompt_tst = prompt + f"Sentence: {X_test[i].strip()[0:thresh]}"

            if "t5" in model.name_or_path:
                encodings_dict = tokenizer(
                    prompt_tst, truncation=True, return_tensors="pt"
                )
            else:
                encodings_dict = tokenizer(
                    prompt_tst, max_length=2048, truncation=True, return_tensors="pt"
                )
            input_ids = encodings_dict.input_ids
            ys_b_test = torch.where(
                (input_ids == pos_token_id_space) | (input_ids == neg_token_id_space),
                input_ids,
                torch.tensor(-100),
            )

            idxs_test = torch.where(ys_b_test != -100)[1]
            idxs_np_test = idxs_test.numpy()

            embed_dp_tst = model(
                input_ids=input_ids.cuda(), return_dict=True, output_hidden_states=True
            )
            embed_comp = embed_dp_tst.hidden_states[hidden_id].squeeze()
            embed_dp = torch.mean(embed_comp[idxs_np_test[-1] + 1 :, :], axis=0)
            embed_tst.append(embed_dp.detach().cpu())
            torch.cuda.empty_cache()

    embed_np_tst_cor = [tensor.numpy() for tensor in embed_tst]
    X_tst_embed_cor = np.stack(embed_np_tst_cor)
    return X_tr_embed_cor, X_tst_embed_cor, y_tr_shuffle


# Perform PCA and then check accuracy
def compute_pca(n_comp: int, X_tr_embed_cor: np.array, X_tst_embed_cor: np.array):
    pca = PCA(n_components=n_comp)
    pca.fit(X_tr_embed_cor)
    X_tr_pca_cor = pca.transform(X_tr_embed_cor)
    X_tst_pca_cor = pca.transform(X_tst_embed_cor)

    X_tr_pca_cor_mean = X_tr_pca_cor.mean(axis=0)
    X_tr_pca_cor_m0 = X_tr_pca_cor - X_tr_pca_cor_mean
    X_tst_pca_cor_m0 = X_tst_pca_cor - X_tr_pca_cor_mean

    cov_X_cor = np.cov(X_tr_pca_cor_m0, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_X_cor)
    D = np.diag(1.0 / np.sqrt(eigenvalues))
    X_tr_pca_cor_white = (eigenvectors @ D @ eigenvectors.T @ X_tr_pca_cor_m0.T).T
    X_tst_pca_cor_white = (eigenvectors @ D @ eigenvectors.T @ X_tst_pca_cor_m0.T).T

    return X_tr_pca_cor_white, X_tst_pca_cor_white


def compute_pca_non_corrupt(
    n_comp: int,
    X_tr_embed: np.array,
    X_tst_embed: np.array,
):
    pca = PCA(n_components=n_comp)
    pca.fit(X_tr_embed)
    X_tr_pca = pca.transform(X_tr_embed)
    X_tst_pca = pca.transform(X_tst_embed)

    X_tr_pca_mean = X_tr_pca.mean(axis=0)
    X_tr_pca_m0 = X_tr_pca - X_tr_pca_mean
    X_tst_pca_m0 = X_tst_pca - X_tr_pca_mean

    cov_X = np.cov(X_tr_pca_m0, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_X)
    D = np.diag(1.0 / np.sqrt(eigenvalues + 1e-8))
    X_tr_pca_white = (eigenvectors @ D @ eigenvectors.T @ X_tr_pca_m0.T).T
    X_tst_pca_white = (eigenvectors @ D @ eigenvectors.T @ X_tst_pca_m0.T).T

    return X_tr_pca_white, X_tst_pca_white


def get_embeds_stream(
    tokenizer: Tokenizer,
    model: torch.nn.Module,
    X_test: List,
    x_k: List,
    y_k: List,
    k: int,
    thresh: int = 100,
    seed: int = 42,
):
    embed_list = []
    embed_id = -1

    # Obtain train embeddings, uncorrupted
    with torch.no_grad():
        for txt in x_k:
            train_txt = [txt.strip()[0:thresh]]
            encodings_dict = tokenizer(train_txt, truncation=True)
            embed = model(
                input_ids=torch.tensor(encodings_dict["input_ids"]).cuda(),
                return_dict=True,
                output_hidden_states=True,
            )
            embed_all = embed.hidden_states[embed_id]
            txt_embed = torch.mean(embed_all, axis=1).squeeze().detach().cpu()
            embed_list.append(txt_embed)

    embed_np_tr = [tensor.numpy() for tensor in embed_list]
    X_tr_embed = np.stack(embed_np_tr)

    embed_tst = []
    # Obtain test embeddings, uncorrupted
    with torch.no_grad():
        for txt in X_test:
            test_txt = [txt.strip()[0:thresh]]
            encodings_dict = tokenizer(test_txt, truncation=True)
            embed = model(
                input_ids=torch.tensor(encodings_dict["input_ids"]).cuda(),
                return_dict=True,
                output_hidden_states=True,
            )
            embed_all = embed.hidden_states[embed_id]
            txt_embed = torch.mean(embed_all, axis=1).squeeze().detach().cpu()
            embed_tst.append(txt_embed)
    embed_np_tst = [tensor.numpy() for tensor in embed_tst]
    X_tst_embed = np.stack(embed_np_tst)
    return X_tr_embed, X_tst_embed, y_k


def get_loo_sequence(
    embed_tokenizer,
    embed_model,
    X_test,
    X_train_subset,
    y_train_subset,
    y_test,
    k,
    text_threshold,
    seed,
    num_pca_components,
):
    # get embeddings
    (
        X_tr_embed_cor,
        X_tst_embed_cor,
        y_tr_shuffle,
    ) = get_embeds_loo(
        embed_tokenizer,
        embed_model,
        X_test,
        X_train_subset,
        y_train_subset,
        k,
        thresh=text_threshold,
        seed=seed,
    )

    # compute pca
    X_tr_pca_cor_white, X_tst_pca_cor_white = compute_pca(
        num_pca_components,
        X_tr_embed_cor,
        X_tst_embed_cor,
    )

    X_tr_pca_cor_white_torch = torch.from_numpy(X_tr_pca_cor_white).float()
    X_tst_pca_cor_white_torch = torch.from_numpy(X_tst_pca_cor_white).float()
    y_tr_shuffle_torch = torch.Tensor(y_tr_shuffle).float()
    y_test_torch = torch.Tensor(y_test).float()
    return (
        X_tr_pca_cor_white_torch,
        X_tst_pca_cor_white_torch,
        y_tr_shuffle_torch,
        y_test_torch,
    )


def get_vanilla_sequence(
    embed_tokenizer,
    embed_model,
    X_test,
    X_train_subset,
    y_train_subset,
    y_test,
    k,
    text_threshold,
    seed,
    num_pca_components,
):
    # get embeddings
    (
        X_tr_embed_cor,
        X_tst_embed_cor,
        y_tr_shuffle,
    ) = get_embeds_vanilla(
        embed_tokenizer,
        embed_model,
        X_test,
        X_train_subset,
        y_train_subset,
        k,
        thresh=text_threshold,
        seed=seed,
    )

    # run pca
    X_tr_pca_cor_white, X_tst_pca_cor_white = compute_pca(
        num_pca_components,
        X_tr_embed_cor,
        X_tst_embed_cor,
    )

    X_tr_pca_cor_white_torch = torch.from_numpy(X_tr_pca_cor_white).float()
    X_tst_pca_cor_white_torch = torch.from_numpy(X_tst_pca_cor_white).float()
    y_tr_shuffle_torch = torch.Tensor(y_tr_shuffle).float()
    y_test_torch = torch.Tensor(y_test).float()

    return (
        X_tr_pca_cor_white_torch,
        X_tst_pca_cor_white_torch,
        y_tr_shuffle_torch,
        y_test_torch,
    )


def get_stream_sequence(
    embed_tokenizer,
    embed_model,
    X_test,
    X_train_subset,
    y_train_subset,
    y_test,
    k,
    text_threshold,
    seed,
    num_pca_components,
):
    # get embeddings
    X_tr_embed_cor, X_tst_embed_cor, _ = get_embeds_stream(
        embed_tokenizer,
        embed_model,
        X_test,
        X_train_subset,
        y_train_subset,
        k,
        thresh=text_threshold,
    )
    # run pca
    (
        X_tr_pca_cor_white,
        X_tst_pca_cor_white,
    ) = compute_pca(
        num_pca_components,
        X_tr_embed_cor,
        X_tst_embed_cor,
    )
    X_tr_pca_cor_white_torch = torch.from_numpy(X_tr_pca_cor_white).float()
    X_tst_pca_cor_white_torch = torch.from_numpy(X_tst_pca_cor_white).float()
    y_tr_shuffle_torch = torch.Tensor(y_train_subset).float()
    y_test_torch = torch.Tensor(y_test).float()

    return (
        X_tr_pca_cor_white_torch,
        X_tst_pca_cor_white_torch,
        y_tr_shuffle_torch,
        y_test_torch,
    )


def get_stream_sequence_audio(
    embed_tokenizer,
    embed_model,
    X_test,
    X_train_subset,
    y_train_subset,
    y_test,
    k,
    text_threshold,
    seed,
    num_pca_components,
):
    # get embeddings
    X_tr_embed_cor, X_tst_embed_cor, y_tr_shuffle = get_embeds_stream_audio(
        embed_tokenizer,
        embed_model,
        X_test,
        X_train_subset,
        y_train_subset,
        k,
        seed=seed,
    )
    # run pca
    (
        X_tr_pca_cor_white,
        X_tst_pca_cor_white,
    ) = compute_pca(
        num_pca_components,
        X_tr_embed_cor.cpu(),
        X_tst_embed_cor.cpu(),
    )
    X_tr_pca_cor_white_torch = torch.from_numpy(X_tr_pca_cor_white).float()
    X_tst_pca_cor_white_torch = torch.from_numpy(X_tst_pca_cor_white).float()
    y_tr_shuffle_torch = torch.Tensor(y_tr_shuffle).float()
    y_test_torch = torch.Tensor(y_test).float()

    return (
        X_tr_pca_cor_white_torch,
        X_tst_pca_cor_white_torch,
        y_tr_shuffle_torch,
        y_test_torch,
    )


def get_stream_sequence_image(
    embed_tokenizer,
    embed_model,
    X_test,
    X_train_subset,
    y_train_subset,
    y_test,
    k,
    text_threshold,
    seed,
    num_pca_components,
):
    # get embeddings
    X_tr_embed_cor, X_tst_embed_cor, y_tr_shuffle = get_embeds_stream_image(
        embed_tokenizer,
        embed_model,
        X_test,
        X_train_subset,
        y_train_subset,
        k,
        seed=seed,
    )

    # compute pca
    (
        X_tr_pca_cor_white,
        X_tst_pca_cor_white,
    ) = compute_pca(
        num_pca_components,
        X_tr_embed_cor.cpu(),
        X_tst_embed_cor.cpu(),
    )
    X_tr_pca_cor_white_torch = torch.from_numpy(X_tr_pca_cor_white).float()
    X_tst_pca_cor_white_torch = torch.from_numpy(X_tst_pca_cor_white).float()
    y_tr_shuffle_torch = torch.Tensor(y_tr_shuffle).float()
    y_test_torch = torch.Tensor(y_test).float()

    return (
        X_tr_pca_cor_white_torch,
        X_tst_pca_cor_white_torch,
        y_tr_shuffle_torch,
        y_test_torch,
    )
