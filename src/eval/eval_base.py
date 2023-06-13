import argparse
import logging
import os
import pickle
import warnings
from typing import List

import torch
from eval_utils import generate_in_context_example, get_template, load_data, load_model
from tokenizers import Tokenizer
from tqdm import tqdm

logging.getLogger("transformers").setLevel(logging.CRITICAL)


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="tqdm")


def evaluate_original_or_tuned(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    dataset: str,
    data_path: str,
    input_key: str,
    seed: int = 42,
    prompt_format: str = "sentence_label",
    random_seed: int = None,
    k_range: List = [4, 8, 16, 32, 48, 64, 96, 128],
    text_threshold: int = 100,
) -> dict:
    """
    Evaluates the performance of an original or tuned model on a given dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        tokenizer (Tokenizer): The tokenizer used for tokenizing text inputs.
        dataset (str): Name of the dataset.
        data_path (str): Path to the data.
        input_key (str): The input key for retrieving the relevant data from the dataset.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        prompt_format (str, optional): The format of prompts. Defaults to "sentence_label".
        random_seed (int, optional): Random seed.
        k_range (List[int], optional): List of k-values for evaluation. Defaults to [4, 8, 16, 32, 48, 64, 96, 128].
        text_threshold (int, optional): Threshold for truncation length of input sequence . Defaults to 100.

    Returns:
        dict: A dictionary containing the evaluation results, including precision, recall, and F1-score for each k-value.
    """
    map_label = {0: "negative", 1: "positive"}
    postive_token_id_no_space = tokenizer("positive").input_ids[0]
    negative_token_id_no_space = tokenizer("negative").input_ids[0]
    positive_token_id_space = tokenizer(" positive").input_ids[0]
    negative_token_id_space = tokenizer(" negative").input_ids[0]

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
            train_template, test_template = get_template(prompt_format)
            in_context_example, ys = generate_in_context_example(
                X_train[0:k],
                y_train[0:k],
                template=train_template,
                seed=(seed if random_seed is None else random_seed),
                order=None,
                text_threshold=text_threshold,
            )

            for _, (text, label) in tqdm(enumerate(zip(X_test, y_test))):
                current_sample = test_template.format(
                    sentence=text[0:text_threshold].strip(), label=""
                )

                prompt = f"{in_context_example}{current_sample}"
                input_ids = tokenizer(
                    prompt, max_length=2048, truncation=True, return_tensors="pt"
                ).input_ids

                sample_outputs = model.generate(
                    input_ids.cuda(),
                    do_sample=False,
                    temperature=0.0,
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

                pred_text = tokenizer.decode(
                    sample_outputs["sequences"][0], skip_special_tokens=True
                )

                logits = torch.softmax(sample_outputs["scores"][0], axis=-1)
                pos_score_space = logits[:, positive_token_id_space].item()
                pos_score_no_space = logits[:, postive_token_id_no_space].item()
                neg_score_space = logits[:, negative_token_id_space].item()
                neg_score_no_space = logits[:, negative_token_id_no_space].item()

                pred_label = pred_text.split(":")[-1].strip()

                # append results
                if label in map_label:
                    gt_label.append(map_label[label])
                else:
                    gt_label.append(label)

                predicted_label.append(pred_label)
                original_text.append(text)
                predicted_text.append(pred_text)
                predicted_scores.append(
                    (
                        pos_score_space,
                        pos_score_no_space,
                        neg_score_space,
                        neg_score_no_space,
                    )
                )
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


def eval_base(
    data_path: str,
    model_name="EleutherAI/gpt-neo-1.3B",
    path_to_finetuned_base_model: str = None,
    dataset: str = "sms_spam",
    key: str = "text",
    prompt_format: str = "sentence_label",
    k_range: List = [4, 8, 16, 32, 48, 64, 96, 128],
    save_dir: str = "./outputs",
    text_threshold: int = 100,
    seeds: List = [42, 69, 128, 512, 1024],
):
    """
    Evaluate the performance of a base model on a given dataset.

    Args:
        data_path (str): Path to the data.
        model_name (str, optional): Name of the base model. Defaults to "EleutherAI/gpt-neo-1.3B".
        path_to_finetuned_base_model (str, optional): Path to the finetuned base model. Defaults to None.
        dataset (str, optional): Name of the dataset. Defaults to "sms_spam".
        key (str, optional): The key for retrieving the relevant data from the dataset. Defaults to "text".
        prompt_format (str, optional): The format of prompts. Defaults to "sentence_label".
        k_range (List[int], optional): List of k-values for evaluation. Defaults to [4, 8, 16, 32, 48, 64, 96, 128].
        save_dir (str, optional): Directory to save the outputs. Defaults to "./outputs".
        text_threshold (int, optional): Text threshold for determining when to process text inputs. Defaults to 100.
        seeds (List[int], optional): List of random seeds. Defaults to [42, 69, 128, 512, 1024].

    Returns:
        None: This function does not return any value. The evaluation results are saved in `save_dir`.
    """
    ### Loading base model
    print("loading base model...")
    if path_to_finetuned_base_model is not None:
        print("loading fine-tuned model")
        model, _, tokenizer = load_model(
            path_to_finetuned_model=path_to_finetuned_base_model, model_name=model_name
        )
    else:
        _, model, tokenizer = load_model(
            path_to_finetuned_model=path_to_finetuned_base_model, model_name=model_name
        )

    print("evaling base model...")
    final_results_base = {}
    for rs in [9]:
        for seed in seeds:
            results = evaluate_original_or_tuned(
                model,
                tokenizer,
                dataset,
                data_path=data_path,
                input_key=key,
                seed=seed,
                prompt_format=prompt_format,
                random_seed=None,
                k_range=k_range,
                text_threshold=text_threshold,
            )
            final_results_base[seed] = results

    model_name_split = model_name.split("/")[-1]
    save_path_base = f"{save_dir}/{dataset}/{model_name_split}/base/"
    if not os.path.exists(save_path_base):
        os.makedirs(save_path_base)

    file_name = f"{save_path_base}/base_tt{text_threshold}.pkl"
    pickle.dump(final_results_base, open(file_name, "wb"))


if __name__ == "__main__":
    # import argparse

    parser = argparse.ArgumentParser(
        description="Helper script for evaluating base model ICL performance."
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
        default="[4, 8, 16, 32, 48, 64, 96, 128]",
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
        "--corrupted_embeds",
        type=bool,
        default=True,
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

    eval_base(
        model_name=args.base_model_name,
        path_to_finetuned_base_model=args.path_to_finetuned_base_model,
        dataset=args.dataset,
        key=args.key,
        data_path=args.data_path,
        prompt_format=args.prompt_format,
        k_range=eval(args.k_range),
        save_dir=args.save_dir,
        text_threshold=args.text_threshold,
        seeds=eval(args.seeds),
    )
