"""Canonical action mapping and deterministic, rank-local training batches."""
import torch
from torch.utils.data import Sampler

JOINT14_SLOTS = tuple(range(6)) + (16,) + tuple(range(40, 46)) + (56,)


def joint14_to_80(value):
    if value.shape[-1] != 14 or not torch.isfinite(value).all():
        raise ValueError("Expected finite joint14")
    output = value.new_zeros((*value.shape[:-1], 80))
    output[..., list(JOINT14_SLOTS)] = value
    return output


def joint80_to_14(value):
    if value.shape[-1] != 80:
        raise ValueError("Expected canonical80")
    return value[..., list(JOINT14_SLOTS)]


def dimension_padding(device=None):
    mask = torch.ones(80, dtype=torch.bool, device=device)
    mask[list(JOINT14_SLOTS)] = False
    return mask


class EpochBatchSampler(Sampler):
    """Shard exactly once; drop incomplete global accumulation groups.

    The trainer owns epoch and next_batch. Workers never alter this progress;
    constructing an iterator after resume reproduces the remaining indices.
    """

    def __init__(self, size, batch_size, *, rank=0, world_size=1, seed=42,
                 accumulation=1, epoch=0, start_batch=0):
        if min(size, batch_size, world_size, accumulation) <= 0 or not 0 <= rank < world_size:
            raise ValueError("Invalid batch topology")
        self.size, self.batch_size = size, batch_size
        self.rank, self.world_size = rank, world_size
        self.seed, self.accumulation = seed, accumulation
        self.epoch, self.start_batch = epoch, start_batch
        self.total_batches = size // (batch_size * world_size)
        self.total_batches -= self.total_batches % accumulation
        if self.total_batches == 0:
            raise ValueError("Not enough windows for one complete global optimizer update")
        if not 0 <= start_batch <= self.total_batches or start_batch % accumulation:
            raise ValueError("Resume batch must be an optimizer-update boundary")

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=generator).tolist()
        width = self.batch_size * self.world_size
        for batch in range(self.start_batch, self.total_batches):
            begin = batch * width + self.rank * self.batch_size
            yield indices[begin:begin + self.batch_size]

    def __len__(self):
        return self.total_batches - self.start_batch
