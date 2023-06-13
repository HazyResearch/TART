import math

import numpy as np
import torch


class DataSampler:
    def __init__(self, n_dims: int):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name: str, n_dims: int, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "nl": NLSyntheticSampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


class NLSyntheticSampler(DataSampler):
    def __init__(self, n_dims: int, bias: float = None, scale: float = None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(
        self, n_points: int, b_size: int, n_dims_truncated: int = None, seeds=None
    ):
        xs_b = np.random.choice([-1, 1], (b_size, n_points, self.n_dims))
        # set sample_sentence to a tensor of type double
        xs_b = torch.tensor(xs_b, dtype=torch.float32)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = -1
        return xs_b, None


class GaussianSampler(DataSampler):
    def __init__(self, n_dims: int, bias: float = None, scale: float = None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b, None
