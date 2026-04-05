import math
import torch
import gc 
import time 
from typing import Optional
from torch import nn 
from torch.utils.flop_counter import FlopCounterMode
from contextlib import nullcontext

from dataclasses import dataclass

@dataclass
class Config:
    n_embd:int
    block_size:int 
    n_layer:int
    

@dataclass
class GPT:
    model: nn.Module
    config: Config
    training:bool

def start_memory_tracking():
    """Initialize GPU memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    else:
        print("This notebook is intended for CUDA GPUs but CUDA is not available.")


def print_memory_usage():
    max_gpu_memory = torch.cuda.max_memory_allocated() / (1024**3)  # Convert bytes to GB
    print(f"Maximum GPU memory allocated: {max_gpu_memory:.1f} GB")


def cleanup(device):
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)  # some buffer time to allow memory to clear
    torch.cuda.reset_peak_memory_stats()
    max_memory_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
    print(f"Maximum GPU memory allocated: {max_memory_allocated:.1f} GB")


def num_parameters(module: nn.Module, requires_grad: Optional[bool] = None) -> int:
    return sum(
        p.numel()
        for p in module.parameters()
        if requires_grad is None or p.requires_grad == requires_grad
    )


def flops_per_param(config: Config, n_params: int) -> int:
    flops_per_token = (
        2 * n_params
    )  # each parameter is used for a MAC (2 FLOPS) per network operation
    # this assumes that all samples have a fixed length equal to the block size
    # which is most likely false during finetuning
    flops_per_seq = flops_per_token * config.block_size
    attn_flops_per_seq = (
        config.n_layer * 2 * 2 * (config.n_embd * (config.block_size**2))
    )
    return flops_per_seq + attn_flops_per_seq


def measure_flops(model: GPT, x: torch.Tensor) -> int:
    """Measures real FLOPs for HFU"""
    flop_counter = FlopCounterMode(model, display=False)
    ctx = nullcontext() if model.training else torch.no_grad()
    with ctx, flop_counter:
        y = model(x)
        if model.training:
            y.sum().backward()
    return flop_counter.get_total_flops()


def estimate_flops(model: GPT) -> int:
    """Measures estimated FLOPs for MFU.

    Refs:
        * https://ar5iv.labs.arxiv.org/html/2205.05198#A1
        * https://ar5iv.labs.arxiv.org/html/2204.02311#A2
    """
    # using all parameters for this is a naive over estimation because not all model parameters actually contribute to
    # this FLOP computation (e.g. embedding, norm). For this reason, the result will be higher by a fixed percentage
    # (~10%) compared to the measured FLOPs, making those lower but more realistic.
    # For a proper estimate, this needs a more fine-grained calculation as in Appendix A of the paper.
    n_trainable_params = num_parameters(model, requires_grad=True)
    trainable_flops = flops_per_param(model.config, n_trainable_params)
    # forward + backward + gradients (assumes no gradient accumulation)
    ops_per_step = 3 if model.training else 1
    n_frozen_params = num_parameters(model, requires_grad=False)
    frozen_flops = flops_per_param(model.config, n_frozen_params)
    # forward + backward
    frozen_ops_per_step = 2 if model.training else 1
    return ops_per_step * trainable_flops + frozen_ops_per_step * frozen_flops



print(f"{torch.utils.data.get_worker_info()=}")

class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example only works with end >= start"
        self.start = start
        self.end = end
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter(range(iter_start, iter_end))

# should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
ds = MyIterableDataset(start=3, end=7)

# Single-process loading
print(list(torch.utils.data.DataLoader(ds, num_workers=2)))