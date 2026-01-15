from typing import Dict

from ...utils import count_tensor_params
from .base_transforms import Transform, DictTransform
import torch

class Normalizer(Transform):
    def __init__(self, mean, std, eps=1e-6):
        self.mean = mean
        self.std = std
        self.eps = eps

    def transform(self, data):
        return (data - self.mean)/(self.std + self.eps)
    
    def inverse_transform(self, data):
        return (data * (self.std + self.eps)) + self.mean

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
    
    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

class UnitGaussianNormalizer(Transform):
    """
    UnitGaussianNormalizer normalizes data to be zero mean and unit std.
    """

    def __init__(self, mean=None, std=None, eps=1e-7, dim=None, mask=None):
        """
        mean : torch.tensor or None
            has to include batch-size as a dim of 1
            e.g. for tensors of shape ``(batch_size, channels, height, width)``,
            the mean over height and width should have shape ``(1, channels, 1, 1)``
        std : torch.tensor or None
        eps : float, default is 0
            for safe division by the std
        dim : int list, default is None
            if not None, dimensions of the data to reduce over to compute the mean and std.

            .. important::

                Has to include the batch-size (typically 0).
                For instance, to normalize data of shape ``(batch_size, channels, height, width)``
                along batch-size, height and width, pass ``dim=[0, 2, 3]``

        mask : torch.Tensor or None, default is None
            If not None, a tensor with the same size as a sample,
            with value 0 where the data should be ignored and 1 everywhere else

        Notes
        -----
        The resulting mean will have the same size as the input MINUS the specified dims.
        If you do not specify any dims, the mean and std will both be scalars.

        Returns
        -------
        UnitGaussianNormalizer instance
        """
        super().__init__()

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("mask", mask)

        self.eps = eps
        if mean is not None:
            self.ndim = mean.ndim
        if isinstance(dim, int):
            dim = [dim]
        self.dim = dim
        self.n_elements = 0

    def fit(self, data_batch):
        self.update_mean_std(data_batch)

    def partial_fit(self, data_batch, batch_size=1):
        if 0 in list(data_batch.shape):
            return
        count = 0
        n_samples = len(data_batch)
        while count < n_samples:
            samples = data_batch[count : count + batch_size]
            # print(samples.shape)
            # if batch_size == 1:
            #     samples = samples.unsqueeze(0)
            if self.n_elements:
                self.incremental_update_mean_std(samples)
            else:
                self.update_mean_std(samples)
            count += batch_size

    def update_mean_std(self, data_batch):
        self.ndim = data_batch.ndim  # Note this includes batch-size
        if self.mask is None:
            self.n_elements = count_tensor_params(data_batch, self.dim)
            self.mean = torch.mean(data_batch, dim=self.dim, keepdim=True)
            self.squared_mean = torch.mean(data_batch**2, dim=self.dim, keepdim=True)
            self.std = torch.std(data_batch, dim=self.dim, keepdim=True)
        else:
            batch_size = data_batch.shape[0]
            dim = [i - 1 for i in self.dim if i]
            shape = [s for i, s in enumerate(self.mask.shape) if i not in dim]
            self.n_elements = torch.count_nonzero(self.mask, dim=dim) * batch_size
            self.mean = torch.zeros(shape)
            self.std = torch.zeros(shape)
            self.squared_mean = torch.zeros(shape)
            data_batch[:, self.mask == 1] = 0
            self.mean[self.mask == 1] = (
                torch.sum(data_batch, dim=dim, keepdim=True) / self.n_elements
            )
            self.squared_mean = (
                torch.sum(data_batch**2, dim=dim, keepdim=True) / self.n_elements
            )
            self.std = torch.std(data_batch, dim=self.dim, keepdim=True)

    def incremental_update_mean_std(self, data_batch):
        if self.mask is None:
            n_elements = count_tensor_params(data_batch, self.dim)
            dim = self.dim
        else:
            dim = [i - 1 for i in self.dim if i]
            n_elements = torch.count_nonzero(self.mask, dim=dim) * data_batch.shape[0]
            data_batch[:, self.mask == 1] = 0

        self.mean = (1.0 / (self.n_elements + n_elements)) * (
            self.n_elements * self.mean + torch.sum(data_batch, dim=dim, keepdim=True)
        )
        self.squared_mean = (1.0 / (self.n_elements + n_elements)) * (
            self.n_elements * self.squared_mean
            + torch.sum(data_batch**2, dim=dim, keepdim=True)
        )
        self.n_elements += n_elements

        # 1/(n_i + n_j) * (n_i * sum(x_i^2)/n_i + sum(x_j^2) - (n_i*sum(x_i)/n_i + sum(x_j))^2)
        # = 1/(n_i + n_j)  * (sum(x_i^2) + sum(x_j^2) - sum(x_i)^2 - 2sum(x_i)sum(x_j) - sum(x_j)^2))
        # multiply by (n_i + n_j) / (n_i + n_j + 1) for unbiased estimator
        self.std = torch.sqrt(self.squared_mean - self.mean**2) * self.n_elements / (self.n_elements - 1)

    def transform(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def inverse_transform(self, x):
        return x * (self.std + self.eps) + self.mean

    def forward(self, x):
        return self.transform(x)

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        return self

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        return self

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    @classmethod
    def from_dataset(cls, dataset, dim=None, keys=None, mask=None):
        """Return a dictionary of normalizer instances, fitted on the given dataset

        Parameters
        ----------
        dataset : pytorch dataset
            each element must be a dict {key: sample}
            e.g. {'x': input_samples, 'y': target_labels}
        dim : int list, default is None
            * If None, reduce over all dims (scalar mean and std)
            * Otherwise, must include batch-dimensions and all over dims to reduce over
        keys : str list or None
            if not None, a normalizer is instanciated only for the given keys
        """
        for i, data_dict in enumerate(dataset):
            if not i:
                if not keys:
                    keys = data_dict.keys()
        instances = {key: cls(dim=dim, mask=mask) for key in keys}
        for i, data_dict in enumerate(dataset):
            for key, sample in data_dict.items():
                if key in keys:
                    instances[key].partial_fit(sample.unsqueeze(0))
        return instances

class DictUnitGaussianNormalizer(DictTransform):
    """DictUnitGaussianNormalizer composes
    DictTransform and UnitGaussianNormalizer to normalize different
    fields of a model output tensor to Gaussian distributions w/
    mean 0 and unit variance.

        Parameters
        ----------
        normalizer_dict : Dict[str, UnitGaussianNormalizer]
            dictionary of normalizers, keyed to fields
        input_mappings : Dict[slice]
            slices of input tensor to grab per field, must share keys with above
        return_mappings : Dict[slice]
            _description_
        """
    def __init__(self, 
                 normalizer_dict: Dict[str, UnitGaussianNormalizer],
                 input_mappings: Dict[str, slice],
                 return_mappings: Dict[str, slice]):
        assert set(normalizer_dict.keys()) == set(input_mappings.keys()), \
            "Error: normalizers and model input fields must be keyed identically"
        assert set(normalizer_dict.keys()) == set(return_mappings.keys()), \
            "Error: normalizers and model output fields must be keyed identically"

        super().__init__(transform_dict=normalizer_dict,
                         input_mappings=input_mappings,
                         return_mappings=return_mappings)
    
    @classmethod
    def from_dataset(cls, dataset, dim=None, keys=None, mask=None):
        """Return a dictionary of normalizer instances, fitted on the given dataset

        Parameters
        ----------
        dataset : pytorch dataset
            each element must be a dict {key: sample}
            e.g. {'x': input_samples, 'y': target_labels}
        dim : int list, default is None
            * If None, reduce over all dims (scalar mean and std)
            * Otherwise, must include batch-dimensions and all over dims to reduce over
        keys : str list or None
            if not None, a normalizer is instanciated only for the given keys
        """
        for i, data_dict in enumerate(dataset):
            if not i:
                if not keys:
                    keys = data_dict.keys()
        instances = {key: cls(dim=dim, mask=mask) for key in keys}
        for i, data_dict in enumerate(dataset):
            for key, sample in data_dict.items():
                if key in keys:
                    instances[key].partial_fit(sample.unsqueeze(0))
        return instances

class MinMaxNormalizer(Transform):
    """
    Min-Max Normalizer (renamed terms; same overall interface):
      - Scales to [0, 1] using per-dimension min & max over `dim`.
      - transform(x)         = (x - data_min) / (data_range + eps)
      - inverse_transform(y) = y * (data_range + eps) + data_min
    """

    def __init__(self, data_min=None, data_max=None, eps=1e-7, dim=None, mask=None, **kwargs):
        """
        Parameters
        ----------
        data_min : torch.Tensor or None
        data_max : torch.Tensor or None
        eps      : float
        dim      : list[int] or int or None
            Dimensions to reduce over when computing min/max (must include batch dim; e.g., [0,2,3,4])
        mask     : torch.Tensor or None
            Same shape as a sample (without batch). 1=use, 0=ignore

        Notes
        -----
        If you previously used (mean/std), please migrate to (data_min/data_max).
        Passing mean/std will raise an error to avoid ambiguity.
        """
        super().__init__()

        if "mean" in kwargs or "std" in kwargs:
            raise ValueError("Use 'data_min'/'data_max' instead of 'mean'/'std' for Min–Max normalization.")

        # register buffers
        self.register_buffer("data_min", data_min)
        self.register_buffer("data_max", data_max)
        self.register_buffer("data_range", None)
        self.register_buffer("mask", mask)

        self.eps = eps
        if isinstance(dim, int):
            dim = [dim]
        self.dim = dim
        self.ndim = None
        self.n_elements = 0  # kept to detect first/next partial_fit calls

        # set range if both provided
        if (data_min is not None) and (data_max is not None):
            self.data_range = data_max - data_min
            self.ndim = data_min.ndim

    # -------------------- fitting --------------------
    def fit(self, data_batch):
        self.update_min_max(data_batch)

    def partial_fit(self, data_batch, batch_size=1):
        if 0 in list(data_batch.shape):
            return
        count = 0
        n_samples = len(data_batch)
        while count < n_samples:
            samples = data_batch[count: count + batch_size]
            if self.n_elements:
                self.incremental_update_min_max(samples)
            else:
                self.update_min_max(samples)
            count += batch_size
            #

    def update_min_max(self, data_batch):
        """Compute min/max over the given batch (with mask support)"""
        self.ndim = data_batch.ndim
        if self.mask is None:
            self.n_elements = count_tensor_params(data_batch, self.dim)
            data_min = torch.amin(data_batch, dim=self.dim, keepdim=True)
            data_max = torch.amax(data_batch, dim=self.dim, keepdim=True)
        else:
            # Broadcast mask to batch: mask shape == sample shape (no batch)
            batch_size = data_batch.shape[0]
            dim_wo_batch = [i - 1 for i in self.dim if i]  # for counting only
            self.n_elements = torch.count_nonzero(self.mask, dim=dim_wo_batch) * batch_size

            mask_b = self.mask.unsqueeze(0).to(dtype=torch.bool, device=data_batch.device)
            min_src = data_batch.masked_fill(~mask_b, float("inf"))
            max_src = data_batch.masked_fill(~mask_b, float("-inf"))
            data_min = torch.amin(min_src, dim=self.dim, keepdim=True)
            data_max = torch.amax(max_src, dim=self.dim, keepdim=True)

        self.data_min = data_min
        self.data_max = data_max
        self.data_range = self.data_max - self.data_min

    def incremental_update_min_max(self, data_batch):
        """Update running min/max with a new batch (with mask support)"""
        if self.mask is None:
            n_elements = count_tensor_params(data_batch, self.dim)
            min_b = torch.amin(data_batch, dim=self.dim, keepdim=True)
            max_b = torch.amax(data_batch, dim=self.dim, keepdim=True)
        else:
            dim_wo_batch = [i - 1 for i in self.dim if i]
            n_elements = torch.count_nonzero(self.mask, dim=dim_wo_batch) * data_batch.shape[0]

            mask_b = self.mask.unsqueeze(0).to(dtype=torch.bool, device=data_batch.device)
            min_src = data_batch.masked_fill(~mask_b, float("inf"))
            max_src = data_batch.masked_fill(~mask_b, float("-inf"))
            min_b = torch.amin(min_src, dim=self.dim, keepdim=True)
            max_b = torch.amax(max_src, dim=self.dim, keepdim=True)

        if self.data_min is None or self.data_max is None:
            self.data_min = min_b
            self.data_max = max_b
        else:
            self.data_min = torch.minimum(self.data_min, min_b)
            self.data_max = torch.maximum(self.data_max, max_b)

        self.n_elements += n_elements
        self.data_range = self.data_max - self.data_min

    # -------------------- transforms --------------------
    def transform(self, x):
        return (x - self.data_min) / (self.data_range + self.eps)

    def inverse_transform(self, x):
        return x * (self.data_range + self.eps) + self.data_min

    def forward(self, x):
        return self.transform(x)

    # -------------------- device helpers --------------------
    def cuda(self):
        if self.data_min is not None: self.data_min = self.data_min.cuda()
        if self.data_max is not None: self.data_max = self.data_max.cuda()
        if self.data_range is not None: self.data_range = self.data_range.cuda()
        if self.mask is not None: self.mask = self.mask.cuda()
        return self

    def cpu(self):
        if self.data_min is not None: self.data_min = self.data_min.cpu()
        if self.data_max is not None: self.data_max = self.data_max.cpu()
        if self.data_range is not None: self.data_range = self.data_range.cpu()
        if self.mask is not None: self.mask = self.mask.cpu()
        return self

    def to(self, device):
        if self.data_min is not None: self.data_min = self.data_min.to(device)
        if self.data_max is not None: self.data_max = self.data_max.to(device)
        if self.data_range is not None: self.data_range = self.data_range.to(device)
        if self.mask is not None: self.mask = self.mask.to(device)
        return self

    # -------------------- dataset helper --------------------
    @classmethod
    def from_dataset(cls, dataset, dim=None, keys=None, mask=None):
        """Return a dict of normalizer instances, fitted on the given dataset"""
        for i, data_dict in enumerate(dataset):
            if not i and not keys:
                keys = data_dict.keys()
        instances = {key: cls(dim=dim, mask=mask) for key in keys}
        for _, data_dict in enumerate(dataset):
            for key, sample in data_dict.items():
                if key in keys:
                    instances[key].partial_fit(sample.unsqueeze(0))
        return instances

    # -------------------- thin compatibility wrappers --------------------
    # If some old code calls these, they still work but point to min/max logic.
    def update_mean_std(self, data_batch):
        return self.update_min_max(data_batch)

    def incremental_update_mean_std(self, data_batch):
        return self.incremental_update_min_max(data_batch)