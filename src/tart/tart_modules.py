from reasoning_module.models import TransformerModel

import torch

from tqdm import tqdm
from typing import List, Dict, Tuple
import numpy as np
from sklearn.decomposition import PCA
from abc import ABC, abstractmethod


class TartReasoningHead:
    def __init__(
        self,
        n_dims: int,
        n_positions: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        n_y: int,
        path_to_pretrained_head: str,
    ):
        self.n_dims = n_dims
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_y = n_y

        self.tart_head = TransformerModel(
            n_dims=n_dims,
            n_positions=n_positions,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            n_y=n_y,
        )

        tweights = torch.load(path_to_pretrained_head)
        self.tart_head.load_state_dict(tweights, strict=False)
        self.tart_head = self.tart_head.cuda()


class TartEmbeddingLayer:
    domain: str
    embed_type: str

    def __init__(
        self,
        embed_model_name: str,
        num_pca_components: int,
    ):
        self.embed_model_name = embed_model_name
        self.num_pca_components = num_pca_components

    def _load_model_tokenizer(self):
        raise NotImplementedError

    def compute_pca_with_whitening(self, X_tr_embed, X_tst_embed):
        pca = PCA(n_components=self.num_pca_components)
        pca.fit(X_tr_embed)
        X_tr_pca_cor = pca.transform(X_tr_embed)
        X_tst_pca_cor = pca.transform(X_tst_embed)

        X_tr_pca_cor_mean = X_tr_pca_cor.mean(axis=0)
        X_tr_pca_cor_m0 = X_tr_pca_cor - X_tr_pca_cor_mean
        X_tst_pca_cor_m0 = X_tst_pca_cor - X_tr_pca_cor_mean

        cov_X_cor = np.cov(X_tr_pca_cor_m0, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_X_cor)
        D = np.diag(1.0 / np.sqrt(eigenvalues))
        X_tr_pca_cor_white = (eigenvectors @ D @ eigenvectors.T @ X_tr_pca_cor_m0.T).T
        X_tst_pca_cor_white = (eigenvectors @ D @ eigenvectors.T @ X_tst_pca_cor_m0.T).T

        return X_tr_pca_cor_white, X_tst_pca_cor_white

    def embed(self):
        raise NotImplementedError

    def get_domain(self):
        return self.domain

    def get_embed_strategy(self):
        return self.embed_type

    def get_embed_model_name(self):
        return self.embed_model_name


class TartEmbeddingLayerAC(ABC):
    _domain: str
    _embed_type: str
    _hf_model_family: str

    def __init__(
        self,
        embed_model_name: str,
        num_pca_components: int,
    ):
        self.embed_model_name = embed_model_name
        self.num_pca_components = num_pca_components

    @abstractmethod
    def _load_model_tokenizer(self):
        pass

    def _compute_pca_with_whitening(self, X_tr_embed, X_tst_embed):
        pca = PCA(n_components=self.num_pca_components)
        pca.fit(X_tr_embed)
        X_tr_pca_cor = pca.transform(X_tr_embed)
        X_tst_pca_cor = pca.transform(X_tst_embed)

        X_tr_pca_cor_mean = X_tr_pca_cor.mean(axis=0)
        X_tr_pca_cor_m0 = X_tr_pca_cor - X_tr_pca_cor_mean
        X_tst_pca_cor_m0 = X_tst_pca_cor - X_tr_pca_cor_mean

        cov_X_cor = np.cov(X_tr_pca_cor_m0, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_X_cor)
        D = np.diag(1.0 / np.sqrt(eigenvalues))
        X_tr_pca_cor_white = (eigenvectors @ D @ eigenvectors.T @ X_tr_pca_cor_m0.T).T
        X_tst_pca_cor_white = (eigenvectors @ D @ eigenvectors.T @ X_tst_pca_cor_m0.T).T

        return X_tr_pca_cor_white, X_tst_pca_cor_white

    @abstractmethod
    def embed(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @property
    def domain(self) -> str:
        return self._domain

    @property
    def embed_type(self) -> str:
        return self._embed_type

    @property
    def _hf_model_family(self) -> str:
        return self._hf_model_family

    @property
    def _model_name(self) -> str:
        return self._model_name


class Tart:
    def __init__(
        self,
        domain: str,
        embed_model_name: str,
        path_to_pretrained_head: str,
        tart_head_config: dict,
        embed_method: str = "stream",
        num_pca_components: int = None,
        path_to_finetuned_embed_model: str = None,
        cache_dir: str = None,
    ):
        from .registry import EMBEDDING_REGISTRY_AC, DOMAIN_REGISTRY

        assert embed_method in [
            "loo",
            "vanilla",
            "stream",
        ], "embed_method not valid. Must be one of: loo, base, stream"

        assert (
            domain in DOMAIN_REGISTRY["supported"]
        ), "domain not valid. Must be one of: text, audio, image"

        self.embed_method = embed_method
        self.config = tart_head_config

        if num_pca_components is None:
            num_pca_components = self.config["n_dims"]

        self.num_pca_components = num_pca_components
        self.domain = domain
        model_name = embed_model_name.split("/")[-1]
        self.embed_layer = EMBEDDING_REGISTRY_AC[model_name][embed_method](
            embed_model_name=embed_model_name,
            num_pca_components=self.num_pca_components,
            cache_dir=cache_dir,
            path_to_finetuned_embed_model=path_to_finetuned_embed_model,
        )
        self._load_tart_head(path_to_pretrained_head, tart_head_config)

    def set_embed_model(
        self, embed_model_name: str, embed_method: str = "stream", cache_dir: str = None
    ):
        from .registry import EMBEDDING_REGISTRY_AC

        model_name = embed_model_name.split("/")[-1]

        print(f"loading embed model: {embed_model_name} ...")

        self.embed_layer = EMBEDDING_REGISTRY_AC[model_name][embed_method](
            embed_model_name=embed_model_name,
            num_pca_components=self.num_pca_components,
            cache_dir=cache_dir,
        )

    def _load_tart_head(self, path_to_pretrained_head, tart_head_config):
        self.tart_head = TartReasoningHead(
            n_dims=tart_head_config["n_dims"],
            n_positions=tart_head_config["n_positions"],
            n_embd=tart_head_config["n_embd"],
            n_head=tart_head_config["n_head"],
            n_layer=tart_head_config["n_layer"],
            n_y=tart_head_config["n_y"],
            path_to_pretrained_head=path_to_pretrained_head,
        ).tart_head

    def _format_eval_sequence(
        self,
        X_train_hidden: torch.Tensor,
        y_train_hidden: torch.Tensor,
        X_test_hidden: torch.Tensor,
        y_test_hidden: torch.Tensor,
    ) -> List:
        """
        For each test sample in the test set, returns a tuple -- (X,Y).
        X is the embedding representations of the in-context samples prepended to
        the embedding representation of the test sample and Y is a tensor of the labels
        prepending in-context sample labels to the label of the test sample.

        Args:
            X_train_hidden (torch.Tensor): Embedding representation of context for each train
                                            datapoint
            y_train_hidden (torch.Tensor): y label of each train datapoint
            X_test_hidden (torch.Tensor): Embedding representation of context for each train
                                            datapoint
            y_test_hidden (torch.Tensor): y label for each test data point

        Returns:
            List: List of (x,y) pairs
        """
        eval_seqs = []
        for test_idx in range(y_test_hidden.shape[-1]):
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
            eval_seqs.append((xs.cuda(), ys.cuda()))
        return eval_seqs

    def _concatenate_inputs(
        self,
        X_train_hidden: torch.Tensor,
        y_train_hidden: torch.Tensor,
        X_test_hidden: torch.Tensor,
        y_test_hidden: torch.Tensor,
    ) -> List:
        """
        For each test sample in the test set, returns a tuple -- (X,Y).
        X is the embedding representations of the in-context samples prepended to
        the embedding representation of the test sample and Y is a tensor of the labels
        prepending in-context sample labels to the label of the test sample.

        Args:
            X_train_hidden (torch.Tensor): Embedding representation of context for each train
                                            datapoint
            y_train_hidden (torch.Tensor): y label of each train datapoint
            X_test_hidden (torch.Tensor): Embedding representation of context for each train
                                            datapoint
            y_test_hidden (torch.Tensor): y label for each test data point

        Returns:
            List: List of z sequences: [I, T] where I is the in-context samples (x,y) pairs and T is the test sample
        """
        eval_seqs = []
        for test_idx in range(y_test_hidden.shape[-1]):
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
            zs = self.tart_head._combine(xs.cuda(), ys.cuda())
            eval_seqs.append(zs)
        return eval_seqs

    def predict(self, eval_seq: torch.Tensor):
        """
        For a given test sample, returns the prediction of the TART model.
        Takes as input the embedding representations of the in-context samples with labels
        prepended to the test sample
        """
        with torch.no_grad():
            pred = self.tart_head.predict(eval_seq)
        return pred

    def evaluate(
        self,
        X_train: List,
        y_train: List,
        X_test: List,
        y_test: List,
        k: int,
        seed: int,
        **kwargs,  # text_threshold: int,
    ) -> Dict:
        """
        Generates predictions for the test set using the TART model.

        Args:
            X_train (List): List of training samples
            y_train (List): List of training labels
            X_test (List): List of test samples
            y_test (List): List of test labels
            k (int): Number of in-context samples to use for each test sample
            text_threshold (int): Threshold for number of characters in a sample to be considered
            seed (int): Seed for random sampling of in-context samples

        Returns:
            Dict: Dictionary of results containing predictions, ground truth labels, and accuracy
        """
        with torch.no_grad():
            X_train_subset = X_train[0:k]
            y_train_subset = y_train[0:k]
            (
                gt_label,
                predicted_label,
                original_text,
                predicted_text,
                predicted_scores,
            ) = ([], [], [], [], [])
            map_label = {0: "negative", 1: "positive"}
            sigmoid = torch.nn.Sigmoid()

            print("Embedding ICL examples...")
            (
                X_train_hidden,
                X_test_hidden,
                y_train_hidden,
                y_test_hidden,
            ) = self.embed_layer.embed(
                X_test, X_train_subset, y_train_subset, y_test, k, seed=seed, **kwargs
            )

            eval_sequences = self._format_eval_sequence(
                X_train_hidden, y_train_hidden, X_test_hidden, y_test_hidden
            )

            print("Predicting labels...")
            for test_idx, (text, label) in tqdm(enumerate(zip(X_test, y_test))):
                xs, ys = eval_sequences[test_idx]

                # in the case that PCA dimensions and dimensions of reasoning head are different
                if self.config["n_dims"] != self.num_pca_components:
                    db_factor = self.config["n_dims"] // self.num_pca_components
                    xs = torch.cat([xs] * db_factor, dim=-1)

                # outs = self.tart_head(xs.cuda(), ys.cuda())
                outs = self.tart_head(xs, ys)
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

            results = {
                "original_text": original_text,
                "predicted_label": predicted_label,
                "gt_label": gt_label,
                "predicted_scores": predicted_scores,
                "accuracy": sum(
                    [1 if x == y else 0 for x, y in zip(gt_label, predicted_label)]
                )
                / len(gt_label),
            }
        return results
