
import os
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

# The set of links between keypoints that make up the skeleton in the COCO pose model.
SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11],
    [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2],
    [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
]


@dataclass
class Config:
    """
    This class collects together some hyperparameters of the learned Kalman Filer.
    This is here mainly to provide the all in one place. These values are used
    mainly only in the training phase.
    """
    # How much history to use for back-propagation through time.
    seq_len: int = 8
    # Training parameters.
    max_epochs: int = 50
    batch_size: int = 32
    lr: float = 3e-3
    lr_patience: int = 3
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    patience: int = 6
    # Model parameters.
    num_keypoint = 17
    depth: int = 3
    hidden_dim: int = 32
    output_dims: int = 16
    # System parameters.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=42):
    """
    Set the seed for builtin, numpy, and torch random modules. To be called
    before any operation involving randomness to ensure repeatability.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def download_h3wb(folder: str):
    """
    Download the H3WB dataset and extract it into the given folder to disk.
    """
    # TODO


class EarlyStopping:
    """
    This is a very small class encapsulating the early stopping logic. The `step`
    method takes the current metric, and returns either `True` if we should stop
    or `False` if the training should continue. Note that for the metric smaller
    should be considered better, i.e., it should be a loss, not an accuracy.
    """

    def __init__(self, patience: int, init: float | None = None):
        self.patience = patience
        self.best = init
        self.counter = 0

    def step(self, metric: float):
        if self.best is None or metric < self.best:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def compute_nll(preds, covars, targets):
    """
    Compute the negative log-likelihood given predicted and ground-truth
    keypoint locations as well as covariances over time.
    """
    # TODO


def compute_mpjpe(preds, targets):
    """
    Compute the Mean Per-Joint Position Error given predicted and ground-truth
    keypoint locations over time.
    """
    # TODO


def train_epoch(model, loader, optimizer, scaler):
    """
    Perform a single training epoch. Also computes the average training loss
    and mean error over the course of the epoch.
    """
    # Ensure we are in training mode (affects e.g. dropout).
    model.train()
    total_loss = 0
    total_mpjpe = 0
    count = 0
    for x, y, dt in loader:
        x = x.to(model.device, non_blocking=True)
        y = y.to(model.device, non_blocking=True)
        dt = dt.to(model.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(model.device.type):
            preds, covars = model(x, dt)
            loss = compute_nll(preds, covars, y)
            mpjpe = compute_mpjpe(preds, y)
        scaler.scale(loss).backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        total_mpjpe += mpjpe.item()
        count += 1
    return total_loss / count, total_mpjpe / count


def eval_epoch(model, loader):
    """
    Run a single evaluation round over the given loader. This is intended to be
    used after each epoch to evaluate the performance on the validation set.
    """
    model.eval()
    total_loss = 0
    total_mpjpe = 0
    count = 0
    # Disable gradients to save memory and compute.
    with torch.no_grad():
        for x, y, dt in loader:
            x = x.to(model.device, non_blocking=True)
            y = y.to(model.device, non_blocking=True)
            dt = dt.to(model.device, non_blocking=True)
            with torch.autocast(model.device.type):
                preds, covars = model(x, dt)
                loss = compute_nll(preds, covars, y)
                mpjpe = compute_mpjpe(preds, y)
            total_loss += loss.item()
            total_mpjpe += mpjpe.item()
            count += 1
    return total_loss / count, total_mpjpe / count


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
        self, nets_dir: str | None, stat_dir: str | None, stopper: EarlyStopping | None,
        net, optimizer, scheduler, compile=True
    ):
        self.nets_dir = nets_dir
        self.stat_dir = stat_dir
        self.step = -1
        self.stopper = stopper
        self.net = net
        if compile:
            self.net.compile()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.load_checkpoint()

    def save_network(self, step: int, metrics: dict, stopper: EarlyStopping | None, net, optimizer, scheduler):
        if self.nets_dir is not None:
            torch.save(net.state_dict(), f"{self.nets_dir}/{step}.net")
            torch.save(optimizer.state_dict(), f"{self.nets_dir}/{step}.optim")
            torch.save(scheduler.state_dict(), f"{self.nets_dir}/{step}.sched")
        if self.stat_dir is not None:
            torch.save({
                "step": step,
                **metrics,
                "stopper_best": stopper.best if stopper is not None else None,
                "stopper_counter": stopper.counter if stopper is not None else None,
            }, f"{self.stat_dir}/{step}.stat")
        # Update the instance variable to reflect the new checkpoint.
        self.step = step
        self.stopper = stopper
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
            stats = torch.load(f"{self.stat_dir}/{step}.stat")
            self.step = step
            if self.stopper is not None and stats["stopper_best"] is not None:
                self.stopper.best = stats["stopper_best"]
            if self.stopper is not None and stats["stopper_counter"] is not None:
                self.stopper.counter = stats["stopper_counter"]
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


def train_epochs(nets: NetStorage, train, val, max_epochs: int, callback=None):
    """
    Perform a multiple training epochs, recoding the history of both training
    and validation loss in the given log directory. This will train using the
    saved optimizer, model, and learning rate schedule from the provided net
    storage.
    """
    model = nets.net
    optimizer = nets.optimizer
    scheduler = nets.scheduler
    stopper = nets.stopper
    scaler = torch.GradScaler(device=model.device.type)
    for epoch in range(nets.step + 1, max_epochs):
        tr_loss, tr_acc = train_epoch(model, train, optimizer, scaler)
        val_loss, val_acc = eval_epoch(model, val)
        nets.save_network(epoch, {
            "tr_loss": tr_loss, "tr_acc": tr_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        }, stopper, model, optimizer, scheduler)
        scheduler.step(val_loss)
        print(
            f"epoch {epoch}; train: loss {tr_loss} acc {tr_acc}; val: loss {val_loss} acc {val_acc}")
        if callback is not None and callback(val_loss, epoch):
            print("stopping via callback")
            break
        if stopper is not None and stopper.step(val_loss):
            print("early stopping")
            break
    else:
        print("max epoch reached")


def net_storage_in(cfg: Config, nets_dir: str | None, stat_dir: str | None, model, compile=True):
    """
    Initialize or load the net storage from the specified directories. This
    function will take the necessary parameters from the supplied configuration.
    """
    stopper = EarlyStopping(cfg.patience)
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.min_lr)
    return NetStorage(nets_dir, stat_dir, stopper, model, optimizer, scheduler, compile)


def train_epochs_in(
    cfg: Config, nets_dir: str | None, stat_dir: str | None, model, subset=None, testing=False, callback=None
):
    """
    Perform a multiple training epochs. This will initialize the net storage in
    case we are starting a fresh run, and resume the existing run otherwise. The
    optimizer, learning rate schedule and early stopper will be initializer
    according to the passed configuration. The test and train split are generated.
    """
    set_seed(42)
    nets = net_storage_in(cfg, nets_dir, stat_dir, model.to(cfg.device))
    full_train = dataset.CloudCastWindowedDataset(
        "./data/train", cfg.seq_len, cfg.pred_len)
    if testing:
        # This configuration is only for the sanity check, it is not used for the
        # actual training of the models.
        train = torch.utils.data.Subset(full_train, range(100))
        val = train
    else:
        # To eliminate all possible information leakage, the gap is chosen to be the
        # sum of the past and future frames loaded. This will have the effect that no
        # single frame is shared between training and validation set.
        train, val = dataset.train_val_split(
            full_train, cfg.val_block_len, cfg.val_block_cnt, cfg.seq_len + cfg.pred_len)
        if subset is not None:
            sub_len = round(subset * len(train))
            train, _ = torch.utils.data.random_split(
                train, [sub_len, len(train) - sub_len])
            sub_len = round(subset * len(val))
            val, _ = torch.utils.data.random_split(
                val, [sub_len, len(val) - sub_len])
    train_loader = DataLoader(
        train, cfg.batch_size, shuffle=True, drop_last=True, num_workers=8,
        persistent_workers=True, pin_memory=True, prefetch_factor=4)
    val_loader = DataLoader(
        val, cfg.batch_size, shuffle=True, drop_last=True, num_workers=8,
        persistent_workers=True, pin_memory=True, prefetch_factor=4)
    train_epochs(nets, train_loader, val_loader, cfg.max_epochs, callback)


def count_model_params(model):
    """
    This is a simple function that just computes the number of trainable
    parameters of the given model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def reduce_logit(logits):
    """
    Reduce the target to use only 3 cloud types, from the 11 that are in the
    full dataset.
    """
    probs = nn.functional.softmax(logits, dim=2)
    mapping = torch.tensor(CLOUD_TYPE_MAPPING, device=logits.device)
    mapping_mat = nn.functional.one_hot(mapping, 4).float()
    return torch.log(torch.einsum('btihw, io -> btohw', probs, mapping_mat))


def reduce_target(target):
    """
    Reduce the target to use only 3 cloud types, from the 11 that are in the
    full dataset.
    """
    mapping = torch.tensor(CLOUD_TYPE_MAPPING, device=target.device)
    return mapping[target]


def compute_confusion_matrix(logits, targets, num_classes=11):
    """
    Compute the confusion matrix given logits and ground-truth targets.
    """
    # Select the maximum for each pixel. Third dimension is the classes.
    pred = torch.argmax(logits, dim=2)
    combined = targets * num_classes + pred
    counts = torch.bincount(torch.flatten(combined), minlength=num_classes**2)
    return counts.reshape(num_classes, num_classes)


def autoregressive_forecast(ahead: int, expand: int, model, x, ts, ts_next):
    _, T, _, _ = x.shape
    ts = torch.concat([ts, ts_next], dim=1)
    parts = []
    for i in range(0, ahead + expand, ahead):
        logits = model(x, ts[:, i:i+T], ts[:, i+T:i+T+ahead])
        parts.append(logits)
        x = torch.concat([x, torch.argmax(logits, dim=2)], dim=1)[:, -T:, :, :]
    return torch.concat(parts, dim=1)


def test_epoch(ahead: int, extend: int, model, loader):
    """
    Run a single evaluation round over the given loader. This differs from the
    function `eval_epoch` in that it collects metrics also for the individual
    prediction steps, not just the average over them. It also compute confusion
    matrices and metrics using grouped classes.
    """
    model.eval()
    metrics = {
        "loss": 0.0, "acc": 0.0, "conf": torch.zeros(11, 11, device=model.device),
        "rloss": 0.0, "racc": 0.0, "rconf": torch.zeros(4, 4, device=model.device),
    }
    for i in range(ahead + extend):
        metrics[f"loss{i}"] = 0.0
        metrics[f"acc{i}"] = 0.0
        metrics[f"conf{i}"] = torch.zeros(11, 11, device=model.device)
        metrics[f"rloss{i}"] = 0.0
        metrics[f"racc{i}"] = 0.0
        metrics[f"rconf{i}"] = torch.zeros(4, 4, device=model.device)
    count = 0
    criterion = nn.CrossEntropyLoss()
    rcriterion = nn.NLLLoss()
    # Disable gradients to save memory and compute.
    with torch.no_grad():
        for x, y, ts, ts_y in loader:
            x = x.to(model.device, non_blocking=True)
            y = y.to(model.device, non_blocking=True)
            ts = ts.to(model.device, non_blocking=True)
            ts_y = ts_y.to(model.device, non_blocking=True)
            with torch.autocast(model.device.type):
                if extend > 0:
                    logits = autoregressive_forecast(
                        ahead, extend, model, x, ts, ts_y)
                else:
                    logits = model(x, ts, ts_y)
                metrics["loss"] += compute_loss(logits, y, criterion).item()
                metrics["acc"] += compute_accuracy(logits, y)
                metrics["conf"] += compute_confusion_matrix(logits, y)
                for i in range(ahead + extend):
                    l, t = logits[:, i:i+1, :, :, :], y[:, i:i+1, :, :]
                    metrics[f"loss{i}"] += compute_loss(l, t, criterion).item()
                    metrics[f"acc{i}"] += compute_accuracy(l, t)
                    metrics[f"conf{i}"] += compute_confusion_matrix(l, t)
                rlogp = reduce_logit(logits)
                ry = reduce_target(y)
                metrics["rloss"] += compute_loss(rlogp, ry, rcriterion).item()
                metrics["racc"] += compute_accuracy(rlogp, ry)
                metrics["rconf"] += compute_confusion_matrix(rlogp, ry, 4)
                for i in range(ahead + extend):
                    l, t = rlogp[:, i:i+1, :, :, :], ry[:, i:i+1, :, :]
                    metrics[f"rloss{i}"] += \
                        compute_loss(l, t, rcriterion).item()
                    metrics[f"racc{i}"] += compute_accuracy(l, t)
                    metrics[f"rconf{i}"] += compute_confusion_matrix(l, t, 4)
            count += 1
    for k in metrics.keys():
        if not k.startswith("conf") and not k.startswith("rconf"):
            metrics[k] /= count
    return metrics


def test_epochs_in(cfg: Config, nets_dir: str, stat_dir: str, model, extend: int = 0):
    """
    Perform a run through the test set and return the metrics.
    """
    set_seed(42)
    if not isinstance(model, CopyModel):
        nets = net_storage_in(cfg, nets_dir, stat_dir, model.to(cfg.device))
        nets.load_best_checkpoint()
        model = nets.net
    test = dataset.CloudCastWindowedDataset(
        "./data/test", cfg.seq_len, cfg.pred_len + extend)
    loader = DataLoader(
        test, cfg.batch_size, shuffle=True, drop_last=True, num_workers=8,
        persistent_workers=True, pin_memory=True, prefetch_factor=4)
    return test_epoch(cfg.pred_len, extend, model, loader.W
