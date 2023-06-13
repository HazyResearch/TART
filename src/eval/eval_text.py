import argparse
import logging
import os
import pickle
import sys
import warnings
from typing import List

import torch
from eval_utils import load_data, load_model
from models import TransformerModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(f"{os.path.dirname(os.getcwd())}/src")

from tart.embed_utils import get_vanilla_sequence, get_loo_sequence, get_stream_sequence

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="tqdm")
logging.getLogger("transformers").setLevel(logging.CRITICAL)


def evaluate_tart(
    inference_head: torch.nn.Module,
    embed_model: torch.nn.Module,
    embed_tokenizer: torch.nn.Module,
    data_path: str,
    dataset: str,
    num_pca_components: int = 8,  # for pca
    seed: int = 42,
    input_key: str = "text",
    k_range: List = [4, 8, 16, 32, 48, 64, 96, 128],
    prompt_format: str = "sentence_label",
    text_threshold: int = 00,
    embed_type: str = None,
    random_seed: int = None,
):
    """
    Evaluate the performance of a TART model on a given dataset.

    Args:
        inference_head (torch.nn.Module): The inference head of the TART model.
        embed_model (torch.nn.Module): The embedding model used by the TART model.
        embed_tokenizer (Tokenizer): The tokenizer used for tokenizing text inputs.
        data_path (str): Path to the data.
        dataset (str): Name of the dataset.
        num_pca_components (int, optional): Number of PCA components for dimensionality reduction. Defaults to 8.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        input_key (str, optional): The input key for retrieving the relevant data from the dataset. Defaults to "image".
        k_range (List[int], optional): List of k-values for evaluation. Defaults to [4, 8, 16, 32, 48, 64, 96, 128].
        text_threshold (int, optional): Text threshold for determining when to process text inputs. Defaults to 100.
        embed_type (str, optional): Embedding type. Defaults to None.

    Returns:
        dict: A dictionary containing the evaluation results. Contains predicted scores and accuracy.
    """
    sigmoid = torch.nn.Sigmoid()
    map_label = {0: "negative", 1: "positive"}

    (X_train, y_train), (X_test, y_test) = load_data(
        data_path, dataset, input_key, seed
    )

    results = {0: {}}

    with torch.no_grad():
        for k in k_range:
            (
                gt_label,
                predicted_label,
                original_text,
                predicted_text,
                predicted_scores,
            ) = ([], [], [], [], [])

            results[0][k] = None
            X_train_subset = X_train[0:k]
            y_train_subset = y_train[0:k]

            if embed_type == "loo":
                print("Using LOO embeddings...")
                sequence_gen_function = get_loo_sequence
            elif embed_type == "base":
                print("Using vanilla embeddings...")
                sequence_gen_function = get_vanilla_sequence
            elif embed_type == "stream":
                print("Using stream embeddings...")
                sequence_gen_function = get_stream_sequence

            (
                X_train_hidden,
                X_test_hidden,
                y_train_hidden,
                y_test_hidden,
            ) = sequence_gen_function(
                embed_tokenizer,
                embed_model,
                X_test,
                X_train_subset,
                y_train_subset,
                y_test,
                k,
                text_threshold=text_threshold,
                seed=seed,
                num_pca_components=num_pca_components,
            )

            for test_idx, (text, label) in tqdm(enumerate(zip(X_test, y_test))):
                xs = torch.cat(
                    [
                        X_train_hidden,
                        X_test_hidden[test_idx, :].unsqueeze(0),
                    ],
                    dim=0,
                ).unsqueeze(0)
                ys = torch.cat(
                    [y_train_hidden, y_test_hidden[test_idx : test_idx + 1]],
                    dim=0,
                ).unsqueeze(0)

                outs = inference_head(xs.cuda(), ys.cuda())
                pred = sigmoid(outs)[0][-1].item()

                if pred >= 0.5:
                    pred_text = "positive"
                    pred_label = "positive"
                else:
                    pred_text = "negative"
                    pred_label = "negative"

                predicted_scores.append(pred)
                predicted_label.append(pred_label)
                original_text.append(text)
                predicted_text.append(pred_text)

                if label in map_label:
                    gt_label.append(map_label[label])
                else:
                    gt_label.append(label)

            results[0][k] = {
                "original_text": original_text,
                "predicted_label": predicted_label,
                "gt_label": gt_label,
                "predicted_scores": predicted_scores,
                "accuracy": sum(
                    [1 if x == y else 0 for x, y in zip(gt_label, predicted_label)]
                )
                / len(gt_label),
            }

            print(f"Accuracy (seed={seed}) @ {k}: {results[0][k]['accuracy']}")

    return results


def eval_tart(
    n_dims: int,
    n_positions: int,
    num_pca_components: int,
    path_to_tart_inference_head: str,
    dataset: str,
    input_key: str,
    data_path: str,
    embed_model_name: str,
    path_to_finetuned_base_model=None,
    prompt_format="sentence_label",
    k_range: List = [32, 48, 64, 96, 128],
    save_dir: str = "./outputs",
    text_threshold: int = 100,
    seeds: List = [42, 69, 128, 512, 1024],
    embed_type=None,
):
    """
    Coordinates the evaluation performance of a TART module on a given dataset.

    Args:
        n_dims (int): Number of dimensions.
        n_positions (int): Number of positions.
        num_pca_components (int): Number of PCA components.
        path_to_tart_inference_head (str): Path to the TART inference head.
        dataset (str): Name of the dataset.
        input_key (str): Input key.
        data_path (str): Path to the data.
        embed_model_name (str): Name of the embedding model.
        path_to_finetuned_base_model (str, optional): Path to the finetuned base model. Defaults to None.
        k_range (List[int], optional): List of k-values for evaluation. Defaults to [4, 8, 16, 32, 48, 64, 96, 128].
        save_dir (str, optional): Directory to save the outputs. Defaults to "./outputs".
        seeds (List[int], optional): List of random seeds. Defaults to [42, 69, 128, 512, 1024].
        text_threshold (int, optional): Text threshold. Defaults to None.
        embed_type (str, optional): Embedding type. Defaults to None.

    Returns:
        None: This function does not return any value. The results are saved in the specified directory.
    """

    ### load in tart reasoing module
    print("loading TART reasoning module...")
    tart_module = TransformerModel(
        n_dims=n_dims, n_positions=n_positions, n_embd=256, n_head=8, n_layer=12, n_y=1
    )
    aweights = torch.load(path_to_tart_inference_head)
    tart_module.load_state_dict(aweights, strict=False)
    tart_module = tart_module.cuda()

    ### Load in base embed model
    print("loading embed model")
    if path_to_finetuned_base_model is not None:
        # base embedding model is a fine-tuned model
        embed_model, _, embed_tokenizer = load_model(
            path_to_finetuned_base_model, model_name=embed_model_name
        )
    else:
        embed_tokenizer = AutoTokenizer.from_pretrained(
            embed_model_name,
            pad_token="<|pad|>",
        )
        embed_tokenizer.truncation_side = "left"
        embed_model = AutoModelForCausalLM.from_pretrained(
            embed_model_name,
            cache_dir="/u/scr/nlp/data/neo/hub",
        ).cuda()
        embed_model.eval()

    ### Call eval function
    final_results_tart = {}
    for rs in [9]:
        for seed in seeds:
            print("evaling trained model")
            results_tart = evaluate_tart(
                inference_head=tart_module,
                embed_model=embed_model,
                embed_tokenizer=embed_tokenizer,
                num_pca_components=num_pca_components,
                seed=seed,
                input_key=input_key,
                data_path=data_path,
                dataset=dataset,
                prompt_format=prompt_format,
                random_seed=seed,
                k_range=k_range,
                text_threshold=text_threshold,
                embed_type=embed_type,
            )
            final_results_tart[seed] = results_tart

        run_id = path_to_tart_inference_head.split("/")[-2]
        checkpoint = path_to_tart_inference_head.split("/")[-1].split(".")[0]

    model_name_split = embed_model_name.split("/")[-1]
    save_path_adaptor = (
        f"{save_dir}/{dataset}/{model_name_split}/tart/{run_id}/{checkpoint}/"
    )

    if not os.path.exists(save_path_adaptor):
        os.makedirs(save_path_adaptor)

    file_name = f"{save_path_adaptor}/TART_{embed_type}.pkl"
    pickle.dump(final_results_tart, open(file_name, "wb"))


if __name__ == "__main__":
    # import argparse

    parser = argparse.ArgumentParser(
        description="Sample eval script for text datasets."
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="EleutherAI/gpt-neo-125m",
    )
    parser.add_argument(
        "--path_to_tart_inference_head",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--path_to_finetuned_base_model",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sms",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="text",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/u/nlp/data/ic-nk/nlp_data_final/",
    )
    parser.add_argument(
        "--prompt_format",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--k_range",
        type=str,
        default="[4, 8, 16, 32, 48, 64, 96, 128]",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="[42, 69, 128, 512, 1024]",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
    )
    parser.add_argument(
        "--n_dims",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--n_positions",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--embed_type",
        type=str,
        default=None,
        # add accepted values
        choices=[
            "base",
            "loo",
            "stream",
        ],
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./outputs",
    )
    parser.add_argument(
        "--text_threshold",
        type=int,
        default=100,
    )
    parser.add_argument("--num_pca_components", type=int, default=8)

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    eval_tart(
        embed_model_name=args.base_model_name,
        path_to_tart_inference_head=args.path_to_tart_inference_head,
        dataset=args.dataset,
        input_key=args.key,
        data_path=args.data_path,
        k_range=eval(args.k_range),
        n_dims=args.n_dims,
        n_positions=args.n_positions,
        num_pca_components=args.num_pca_components,
        save_dir=args.save_dir,
        text_threshold=args.text_threshold,
        seeds=eval(args.seeds),
        embed_type=args.embed_type,
    )
