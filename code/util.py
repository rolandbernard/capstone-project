
import os
import random

import torch
import torch.optim as optim
import numpy as np
import pandas as pd


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# The set of links between keypoints that make up the skeleton in the COCO pose model.
SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11],
    [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2],
    [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
]


def set_seed(seed=42):
    """
    Set the seed for builtin, numpy, and torch random modules. To be called
    before any operation involving randomness to ensure repeatability.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def remove_idx(input: torch.Tensor, idx: torch.Tensor, dim=0) -> torch.Tensor:
    """
    Remove all indices in `idx` from the input tensor at dimension `dim`.
    """
    mask = torch.ones(input.shape[dim], dtype=torch.bool, device=input.device)
    mask[idx] = False
    return torch.index_select(input, dim, torch.nonzero(mask).squeeze())


class NetStorage:
    """
    This class provides some utility methods for recoding training history, as
    well as saving and restoring model weights. This should allow stopping and
    resuming the training process after any epoch.
    """

    @classmethod
    def training_stats(cls, stats_dir: str) -> pd.DataFrame:
        return pd.DataFrame([
            torch.load(f"{stats_dir}/{n}") for n in os.listdir(stats_dir)
        ])

    def __init__(
        self, nets_dir: str | None, stat_dir: str | None, net, optimizer, scheduler, compile=True
    ):
        self.nets_dir = nets_dir
        self.stat_dir = stat_dir
        self.step = -1
        self.net = net
        if compile:
            self.net.compile()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.load_checkpoint()

    def save_network(self, step: int, metrics: dict, net, optimizer, scheduler):
        if self.nets_dir is not None:
            torch.save(net.state_dict(), f"{self.nets_dir}/{step}.net")
            torch.save(optimizer.state_dict(), f"{self.nets_dir}/{step}.optim")
            torch.save(scheduler.state_dict(), f"{self.nets_dir}/{step}.sched")
        if self.stat_dir is not None:
            torch.save({
                "step": step,
                **metrics,
            }, f"{self.stat_dir}/{step}.stat")
        # Update the instance variable to reflect the new checkpoint.
        self.step = step
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler

    def load_best_checkpoint(self):
        if self.stat_dir is not None:
            stats = NetStorage.training_stats(self.stat_dir)
            min_idx = stats["val_loss"].idxmin()
            self.load_checkpoint(stats.loc[min_idx].step)

    def load_checkpoint(self, step=None):
        if self.nets_dir is None or self.stat_dir is None:
            # If any of the directories are missing, recovery is not supported.
            return
        # Make sure the directories exist.
        os.makedirs(self.nets_dir, exist_ok=True)
        os.makedirs(self.stat_dir, exist_ok=True)
        # Figure out the latest checkpoints step count.
        if step is None:
            step = max((int(n.split(".")[0])
                       for n in os.listdir(self.stat_dir)), default=-1)
        # Load if the step exists, otherwise this is a fresh run.
        if step != -1 and os.path.isfile(f"{self.stat_dir}/{step}.stat"):
            self.step = step
            model = torch.load(
                f"{self.nets_dir}/{step}.net", map_location=self.net.device
            )
            self.net.load_state_dict(model)
            self.net.eval()
            optim = torch.load(f"{self.nets_dir}/{step}.optim")
            self.optimizer.load_state_dict(optim)
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.net.device)
            sched = torch.load(f"{self.nets_dir}/{step}.sched")
            self.scheduler.load_state_dict(sched)
            print(f"Loaded checkpoint for step {step}.")


def net_storage_in(nets_dir: str | None, stat_dir: str | None, model, compile=True):
    """
    Initialize or load the net storage from the specified directories. This
    function will take the necessary parameters from the supplied configuration.
    """
    optimizer = optim.Adam(model.extra_parameters(), lr=3e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    return NetStorage(nets_dir, stat_dir, model, optimizer, scheduler, compile)


def count_model_params(model):
    """
    This is a simple function that just computes the number of trainable
    parameters of the given model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
