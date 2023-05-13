from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import numpy as np
import random
from typing import List, Tuple, Any, Type, Optional
import copy


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class SurrogateDataLoader(DataLoader):
    def __init__(self, seed=0, dev='cpu', should_weight_last_sample=False, last_sample=None, use_resampling=False,
                 **kwargs):
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(int(seed))
            kwargs['generator'] = g
            kwargs['worker_init_fn'] = seed_worker
        super().__init__(**kwargs)
        self.seed = seed
        self.device = dev
        self.should_weight_last_sample = should_weight_last_sample
        self.last_sample = last_sample
        self.kwargs = kwargs
        self.data_size = 5
        self.use_resampling = use_resampling

    def __len__(self):
        return super().__len__()

    def __iter__(self):
        batches = super().__iter__()
        if self.should_weight_last_sample:
            weighted_batch: List[Optional[torch.Tensor]] = [None] * self.data_size
            for b in batches:
                for i in range(self.data_size):
                    weighted_batch[i] = torch.cat((b[i], self.last_sample[i])).to(self.device)
                yield weighted_batch[0], weighted_batch[1], weighted_batch[2], weighted_batch[3], weighted_batch[4]
        else:
            for b in batches:
                yield b[0].to(self.device), b[1].to(self.device), b[2].to(self.device), b[3].to(self.device), b[4].to(
                    self.device)

    def make_dataloader(self, seed=0):
        kwargs = copy.copy(self.kwargs)
        dataset = self.kwargs['dataset']
        if dataset.use_sample_weights:
            new_dataset = copy.copy(dataset)
            new_dataset.reset_sample_weights()
            kwargs['dataset'] = new_dataset

        # weights = torch.rand((len(dataset),))
        # sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        # kwargs['sampler'] = sampler
        # kwargs['shuffle'] = False
        if self.use_resampling:
            new_dataset = copy.copy(dataset)
            new_dataset.resample_dataset()
            kwargs['dataset'] = new_dataset

        instance = SurrogateDataLoader(seed=seed, dev=self.device,
                                       should_weight_last_sample=self.should_weight_last_sample,
                                       last_sample=self.last_sample,
                                       use_resampling=self.use_resampling,
                                       **kwargs)
        return instance
