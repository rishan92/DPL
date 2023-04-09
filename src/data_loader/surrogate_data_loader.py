from torch.utils.data import DataLoader
import torch
import numpy as np
import random
from typing import List, Tuple, Any, Type


class SurrogateDataLoader(DataLoader):
    def __init__(self, seed=0, dev='cpu', should_weight_last_sample=False, last_sample=None, **kwargs):
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

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

    def __len__(self):
        return super().__len__()

    def __iter__(self):
        batches = super().__iter__()
        if self.should_weight_last_sample:
            weighted_batch = [None] * 4
            for b in batches:
                for i in range(4):
                    weighted_batch[i] = torch.cat((b[i], self.last_sample[i]))
                yield weighted_batch[0].to(self.device), weighted_batch[1].to(self.device), \
                      weighted_batch[2].to(self.device), weighted_batch[3].to(self.device)
        else:
            for b in batches:
                yield b[0].to(self.device), b[1].to(self.device), b[2].to(self.device), b[3].to(self.device)

    def make_dataloader(self, seed=0):
        instance = SurrogateDataLoader(seed=seed, dev=self.device,
                                       should_weight_last_sample=self.should_weight_last_sample,
                                       last_sample=self.last_sample,
                                       **self.kwargs)
        return instance
