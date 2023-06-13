import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer
from typing import List


def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()


def cross_entropy(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)


def cross_entropy_lm(ys_pred, ys):
    loss_fct = CrossEntropyLoss()
    ys_pred = ys_pred[..., :-1, :].contiguous()
    ys = ys[..., 1:].contiguous()
    ys_pred = ys_pred.view(-1, ys_pred.size(-1))
    loss = loss_fct(ys_pred, ys.view(-1))
    return loss


def cross_entropy_zero_one(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = ys
    return bce_loss(output, target)


def cross_entropy_no_reduction(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = ys
    return bce_loss_no_reduce(output, target)


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()
bce_loss_no_reduce = torch.nn.BCELoss(reduction="none")


class Task:
    def __init__(
        self,
        n_dims: int,
        batch_size: int,
        pool_dict: dict = None,
        seeds: List[int] = None,
    ):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(
    task_name: str,
    n_dims: int,
    batch_size: int,
    pool_dict: dict = None,
    num_tasks: int = None,
    **kwargs,
):
    task_names_to_classes = {
        "probabilistic_logistic_regression": ProbabilisticLogisticRegression,
        "nl": NLSyntheticTask,
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError


class NLSyntheticTask(Task):
    def __init__(
        self,
        n_dims: int,
        batch_size: int,
        pool_dict: dict = None,
        seeds: List[int] = None,
        scale: int = 1,
        weight_multiplier: int = 1,
        variable_noise: bool = False,
        default_word: str = "null",
        n_points: int = None,
        tokenizer_name: str = None,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(NLSyntheticTask, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.weight_multiplier = weight_multiplier
        self.n_dims = n_dims
        self.n_points = n_points
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.positive_token_id_space = self.tokenizer(" positive").input_ids[0]
        self.negative_token_id_space = self.tokenizer(" negative").input_ids[0]
        self.valid_words = []
        for x in self.tokenizer.vocab.keys():
            if (
                len(self.tokenizer(f" {x} sports : ").input_ids) == 4
                and len(self.tokenizer(f"{x} sports : ").input_ids) == 4
            ):
                self.valid_words.append(x)

        if "pythia" not in tokenizer_name:
            self.words = [
                "sports",
                "love",
                "hate",
                "car",
                "school",
                "family",
                "work",
                "sleep",
                "water",
                "tree",
                "fox",
                "train",
                "random",
                "movie",
                "music",
                "book",
                "play",
                "house",
                "spell",
                "bar",
                "jump",
                "park",
                "run",
                "hill",
                "fast",
                "slow",
                "talk",
                "wallet",
                "orange",
                "apple",
                "ball",
                "cat",
            ]

        else:
            self.words = [
                "love",
                "car",
                "school",
                "family",
                "work",
                "sleep",
                "water",
                "tree",
                "fox",
                "train",
                "random",
                "movie",
                "music",
                "book",
                "play",
                "house",
                "bar",
                "jump",
                "park",
                "run",
                "hill",
                "fast",
                "slow",
                "talk",
                "orange",
                "apple",
                "ball",
                "cat",
            ]

        self.label_words = {0: "negative", 1: "positive"}
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.default_word = default_word
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self.vocabulary = list(self._tokenizer.vocab.keys())
        self._tokenizer.truncation_side = "left"
        self.model_name = tokenizer_name

        if pool_dict is None and seeds is None:
            if variable_noise:
                self.w_b = torch.randn(self.b_size, self.n_dims, 1)
                self.w_b = self.w_b * torch.randint(1, 6, (self.b_size, 1, 1))
            else:
                self.w_b = torch.randn(self.b_size, self.n_dims, 1) * weight_multiplier

        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                if variable_noise:
                    self.w_b[i] = torch.randn(
                        self.n_dims, 1, generator=generator
                    ) * torch.randint(1, 11, (self.b_size, 1, 1))

                else:
                    self.w_b[i] = (
                        torch.randn(self.n_dims, 1, generator=generator)
                        * weight_multiplier
                    )
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def _construct_sequences(self, xs_b, ys_b):
        batch = []
        for b_idx in range(xs_b.shape[0]):
            sequence = []
            for p_idx in range(xs_b.shape[1]):
                word = []
                for d_idx in range(xs_b.shape[2]):
                    if xs_b[b_idx, p_idx, d_idx] == 1:
                        word.append(self.words[d_idx])
                    else:
                        word.append(self.default_word)

                if ys_b[b_idx, p_idx].item() == 0:
                    label_str = self.label_words[0]
                else:
                    label_str = self.label_words[1]
                sequence.append(((" ".join(word)), label_str))
            batch.append(sequence)

        tok_batch = []
        for sequence in batch:
            input_seq = ""
            for sample in sequence:
                input_seq += " ".join([sample[0], sample[1]])
                input_seq += " "
            tokenized_seq = self._tokenizer(input_seq.strip()).input_ids
            tok_batch.append(tokenized_seq)
        return torch.tensor(tok_batch)

    def _construct_sequences_colon(self, xs_b, ys_b):
        batch = []

        for b_idx in range(xs_b.shape[0]):
            sequence = []
            for p_idx in range(xs_b.shape[1]):
                word = []
                for d_idx in range(xs_b.shape[2]):
                    if xs_b[b_idx, p_idx, d_idx] == 1:
                        word.append(self.words[d_idx])
                    else:
                        word.append(self.default_word)

                if ys_b[b_idx, p_idx].item() == 0:
                    label_str = self.label_words[0]
                else:
                    label_str = self.label_words[1]
                sequence.append(((" ".join(word)), label_str))
            batch.append(sequence)

        tok_batch = []
        for sequence in batch:
            input_seq = ""
            for sample in sequence:
                input_seq += " : ".join([sample[0], sample[1]])
                input_seq += " , "
            tokenized_seq = self._tokenizer(input_seq.strip(" , ")).input_ids
            tok_batch.append(tokenized_seq)
        return torch.tensor(tok_batch)

    def evaluate(self, xs_b, w_b=None):
        if w_b is not None:
            self.w_b = w_b * self.weight_multiplier

        w_b = self.w_b.to(xs_b.device)
        probability = torch.sigmoid(self.scale * (xs_b @ w_b)[:, :, 0])
        ys_b = torch.bernoulli(probability)
        nl_batch = self._construct_sequences_colon(xs_b, ys_b)

        ys_b = torch.where(
            (nl_batch == self.positive_token_id_space)
            | (nl_batch == self.negative_token_id_space),
            nl_batch,
            torch.tensor(-100),
        )
        return ys_b, w_b.detach(), nl_batch

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return cross_entropy_no_reduction

    @staticmethod
    def get_training_metric():
        return cross_entropy_lm


class ProbabilisticLogisticRegression(Task):
    def __init__(
        self,
        n_dims: int,
        batch_size: int,
        pool_dict: dict = None,
        seeds: List[int] = None,
        scale: int = 1,
        weight_multiplier: int = 1,
        variable_noise: bool = False,
        n_points: int = None,
        tokenizer_name: str = None,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(ProbabilisticLogisticRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds
        )
        self.scale = scale
        self.weight_multiplier = weight_multiplier

        if pool_dict is None and seeds is None:
            if variable_noise:
                self.w_b = torch.randn(self.b_size, self.n_dims, 1)
                # multiply each row of w_b by a weight sampled from [1, 2, ... 10]
                self.w_b = self.w_b * torch.randint(
                    1, weight_multiplier + 1, (self.b_size, 1, 1)
                )
            else:
                self.w_b = torch.randn(self.b_size, self.n_dims, 1) * weight_multiplier

        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                if variable_noise:
                    self.w_b[i] = torch.randn(
                        self.n_dims, 1, generator=generator
                    ) * torch.randint(1, 11, (self.b_size, 1, 1))

                else:
                    self.w_b[i] = (
                        torch.randn(self.n_dims, 1, generator=generator)
                        * weight_multiplier
                    )
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b, w_b=None):
        if w_b is not None:
            self.w_b = w_b * self.weight_multiplier

        w_b = self.w_b.to(xs_b.device)

        probability = torch.sigmoid(self.scale * (xs_b @ w_b)[:, :, 0])
        ys_b = torch.bernoulli(probability)
        return ys_b, w_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return cross_entropy_no_reduction

    @staticmethod
    def get_training_metric():
        return cross_entropy_zero_one
