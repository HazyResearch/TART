import random

import numpy as np
import torch
from .embed_utils import (
    get_embeds_vanilla,
    get_embeds_loo,
    get_embeds_stream,
    get_embeds_stream_audio,
    get_embeds_stream_image,
)

from .tart_modules import TartEmbeddingLayerAC

from tqdm import tqdm
from transformers import (
    AutoFeatureExtractor,
    AutoModelForCausalLM,
    AutoTokenizer,
    ViTModel,
    WhisperModel,
)

from eval.eval_utils import load_model
from typing import List


class LOOEmbeddingCausalLM(TartEmbeddingLayerAC):
    _domain = "text"
    _embed_type = "loo"
    _hf_model_family = "AutoModelForCausalLM"

    def __init__(
        self,
        embed_model_name: str,
        num_pca_components: int,
        cache_dir: str = None,
        path_to_finetuned_embed_model: str = None,
    ):
        super().__init__(
            embed_model_name=embed_model_name, num_pca_components=num_pca_components
        )
        self.embed_model_name = embed_model_name
        self._load_model_tokenizer(
            embed_model_name, path_to_finetuned_embed_model, cache_dir
        )

    def _load_model_tokenizer(
        self, embed_model_name, path_to_finetuned_embed_model, cache_dir=None
    ):
        if path_to_finetuned_embed_model is not None:
            # base embedding model is a fine-tuned model
            self.embed_model, _, self.embed_tokenizer = load_model(
                path_to_finetuned_embed_model, model_name=embed_model_name
            )
        else:
            self.embed_tokenizer = AutoTokenizer.from_pretrained(
                embed_model_name,
                pad_token="<|pad|>",
                cache_dir=cache_dir,
            )
            self.embed_tokenizer.truncation_side = "left"
            self.embed_model = AutoModelForCausalLM.from_pretrained(
                embed_model_name,
                cache_dir=cache_dir,
            ).cuda()
            self.embed_model.eval()

    def embed(
        self,
        X_test: List,
        X_train_subset: List,
        y_train_subset: List,
        y_test: List,
        k: int,
        text_threshold: int = 100,
        seed: int = 42,
    ):
        # get embeddings
        (
            X_tr_embed_cor,
            X_tst_embed_cor,
            y_tr_shuffle,
        ) = get_embeds_loo(
            self.embed_tokenizer,
            self.embed_model,
            X_test,
            X_train_subset,
            y_train_subset,
            k,
            thresh=text_threshold,
            seed=seed,
        )

        (
            X_tr_pca_cor_white,
            X_tst_pca_cor_white,
        ) = self._compute_pca_with_whitening(X_tr_embed_cor, X_tst_embed_cor)

        return (
            torch.from_numpy(X_tr_pca_cor_white).float(),
            torch.from_numpy(X_tst_pca_cor_white).float(),
            torch.Tensor(y_tr_shuffle).float(),
            torch.Tensor(y_test).float(),
        )


class VanillaEmbeddingCausalLM(TartEmbeddingLayerAC):
    _domain = "text"
    _embed_type = "vanilla"
    _hf_model_family = "AutoModelForCausalLM"

    def __init__(
        self,
        embed_model_name: str,
        num_pca_components: int,
        cache_dir: str = None,
        path_to_finetuned_embed_model: str = None,
    ):
        super().__init__(
            embed_model_name=embed_model_name, num_pca_components=num_pca_components
        )
        self.embed_model_name = embed_model_name
        self._load_model_tokenizer(
            embed_model_name, path_to_finetuned_embed_model, cache_dir
        )

    def _load_model_tokenizer(
        self, embed_model_name, path_to_finetuned_embed_model, cache_dir=None
    ):
        if path_to_finetuned_embed_model is not None:
            # base embedding model is a fine-tuned model
            self.embed_model, _, self.embed_tokenizer = load_model(
                path_to_finetuned_embed_model, model_name=embed_model_name
            )
        else:
            self.embed_tokenizer = AutoTokenizer.from_pretrained(
                embed_model_name,
                pad_token="<|pad|>",
                cache_dir=cache_dir,
            )
            self.embed_tokenizer.truncation_side = "left"
            self.embed_model = AutoModelForCausalLM.from_pretrained(
                embed_model_name,
                cache_dir=cache_dir,
            ).cuda()
            self.embed_model.eval()

    def embed(
        self,
        X_test: List,
        X_train_subset: List,
        y_train_subset: List,
        y_test: List,
        k: int,
        text_threshold: int = 100,
        seed: int = 42,
    ):
        (
            X_tr_embed_cor,
            X_tst_embed_cor,
            y_tr_shuffle,
        ) = get_embeds_vanilla(
            self.embed_tokenizer,
            self.embed_model,
            X_test,
            X_train_subset,
            y_train_subset,
            k,
            thresh=text_threshold,
            seed=seed,
        )

        # run pca
        (
            X_tr_pca_cor_white,
            X_tst_pca_cor_white,
        ) = self._compute_pca_with_whitening(X_tr_embed_cor, X_tst_embed_cor)

        return (
            torch.from_numpy(X_tr_pca_cor_white).float(),
            torch.from_numpy(X_tst_pca_cor_white).float(),
            torch.Tensor(y_tr_shuffle).float(),
            torch.Tensor(y_test).float(),
        )


class StreamEmbeddingCausalLM(TartEmbeddingLayerAC):
    _domain = "text"
    _embed_type = "stream"
    _hf_model_family = "AutoModelForCausalLM"

    def __init__(
        self,
        embed_model_name: str,
        num_pca_components: int,
        cache_dir: str = None,
        path_to_finetuned_embed_model: str = None,
    ):
        super().__init__(
            embed_model_name=embed_model_name, num_pca_components=num_pca_components
        )
        self.embed_model_name = embed_model_name
        self._load_model_tokenizer(
            embed_model_name, path_to_finetuned_embed_model, cache_dir
        )

    def _load_model_tokenizer(
        self, embed_model_name, path_to_finetuned_embed_model, cache_dir=None
    ):
        if path_to_finetuned_embed_model is not None:
            # base embedding model is a fine-tuned model
            self.embed_model, _, self.embed_tokenizer = load_model(
                path_to_finetuned_embed_model, model_name=embed_model_name
            )
        else:
            self.embed_tokenizer = AutoTokenizer.from_pretrained(
                embed_model_name,
                pad_token="<|pad|>",
                cache_dir=cache_dir,
            )
            self.embed_tokenizer.truncation_side = "left"
            self.embed_model = AutoModelForCausalLM.from_pretrained(
                embed_model_name,
                cache_dir=cache_dir,
            ).cuda()
            self.embed_model.eval()

    def embed(
        self,
        X_test: List,
        X_train_subset: List,
        y_train_subset: List,
        y_test: List,
        k: int,
        text_threshold: int = 100,
        seed: int = 42,
    ):
        X_tr_embed_cor, X_tst_embed_cor, y_tr_shuffle = get_embeds_stream(
            self.embed_tokenizer,
            self.embed_model,
            X_test,
            X_train_subset,
            y_train_subset,
            k,
            thresh=text_threshold,
            seed=seed,
        )

        (
            X_tr_pca_cor_white,
            X_tst_pca_cor_white,
        ) = self._compute_pca_with_whitening(X_tr_embed_cor, X_tst_embed_cor)

        return (
            torch.from_numpy(X_tr_pca_cor_white).float(),
            torch.from_numpy(X_tst_pca_cor_white).float(),
            torch.Tensor(y_tr_shuffle).float(),
            torch.Tensor(y_test).float(),
        )


class StreamEmbeddingWhisper(TartEmbeddingLayerAC):
    _domain = "audio"
    _embed_type = "stream"
    _hf_model_family = "WhisperModel"

    def __init__(
        self,
        embed_model_name: str,
        num_pca_components: int,
        cache_dir: str = None,
        path_to_finetuned_embed_model: str = None,
    ):
        super().__init__(
            embed_model_name=embed_model_name, num_pca_components=num_pca_components
        )
        print("loading model")
        self._load_model_tokenizer(
            embed_model_name, path_to_finetuned_embed_model, cache_dir
        )

    def _load_model_tokenizer(
        self, embed_model_name, path_to_finetuned_embed_model, cache_dir=None
    ):
        self.embed_model = WhisperModel.from_pretrained(
            embed_model_name,
            cache_dir=cache_dir,
        ).cuda()
        self.embed_tokenizer = AutoFeatureExtractor.from_pretrained(
            embed_model_name,
            cache_dir=cache_dir,
        )

    def embed(
        self,
        X_test: List,
        X_train_subset: List,
        y_train_subset: List,
        y_test: List,
        k: int,
        seed: int = 42,
    ):
        X_tr_embed_cor, X_tst_embed_cor, y_tr_shuffle = get_embeds_stream_audio(
            self.embed_tokenizer,
            self.embed_model,
            X_test,
            X_train_subset,
            y_train_subset,
            k,
            seed=seed,
        )

        (
            X_tr_pca_cor_white,
            X_tst_pca_cor_white,
        ) = self._compute_pca_with_whitening(
            X_tr_embed_cor.cpu(),
            X_tst_embed_cor.cpu(),
        )

        return (
            torch.from_numpy(X_tr_pca_cor_white).float(),
            torch.from_numpy(X_tst_pca_cor_white).float(),
            torch.Tensor(y_tr_shuffle).float(),
            torch.Tensor(y_test).float(),
        )


class StreamEmbeddingViT(TartEmbeddingLayerAC):
    _domain = "image"
    _embed_type = "stream"
    _hf_model_family = "ViTModel"

    def __init__(
        self,
        embed_model_name: str,
        num_pca_components: int,
        cache_dir: str = None,
        path_to_finetuned_embed_model: str = None,
    ):
        super().__init__(
            embed_model_name=embed_model_name, num_pca_components=num_pca_components
        )
        print("loading model")
        self._load_model_tokenizer(
            embed_model_name, path_to_finetuned_embed_model, cache_dir
        )

    def _load_model_tokenizer(
        self, embed_model_name, path_to_finetuned_embed_model, cache_dir=None
    ):
        self.embed_model = ViTModel.from_pretrained(
            embed_model_name,
            cache_dir=cache_dir,
        ).cuda()
        self.embed_tokenizer = None

    def embed(
        self,
        X_test: List,
        X_train_subset: List,
        y_train_subset: List,
        y_test: List,
        k: int,
        seed: int = 42,
    ):
        X_tr_embed_cor, X_tst_embed_cor, y_tr_shuffle = get_embeds_stream_image(
            self.embed_tokenizer,
            self.embed_model,
            X_test,
            X_train_subset,
            y_train_subset,
            k,
            seed=seed,
        )

        (
            X_tr_pca_cor_white,
            X_tst_pca_cor_white,
        ) = self._compute_pca_with_whitening(
            X_tr_embed_cor.cpu(),
            X_tst_embed_cor.cpu(),
        )

        return (
            torch.from_numpy(X_tr_pca_cor_white).float(),
            torch.from_numpy(X_tst_pca_cor_white).float(),
            torch.Tensor(y_tr_shuffle).float(),
            torch.Tensor(y_test).float(),
        )
