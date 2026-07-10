
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import util
import dataset
import kalman
from camera import Camera
from util import NetStorage


def simulate_kalman_filter(model, fps: torch.Tensor, track: torch.Tensor, cams: list[Camera]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform a simulated run of the Kalman filter on the given tracks. Observations
    are simulated by applying known normal noise to the true projections. The noise
    level itself is randomly generated. Large noise is added for invisible points.
    Predictions before and after update are returned for each time step.
    """
    pass


def compute_loss(pred, covs, gt: torch.Tensor, w_mse: float) -> torch.Tensor:
    """
    Compute the loss between the predictions and the ground truth. Both are
    assumed to be sequences of 3d keypoint detections. The prediction is composed
    of both a mean and a covariance matrix. The loss is a combination a MSE and
    a NLL term.
    """
    pass


def train_epoch(model, loader, raw_data, optimizer, w_mse: float) -> float:
    """
    Perform a single training epoch. Also computes the average training loss
    over the course of the epoch.
    """
    # Ensure we are in training mode.
    model.train()
    total_loss = 0
    count = 0
    for fps, track in loader:
        fps = fps.to(model.device, non_blocking=True)
        track = track.to(model.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred0, covs0, pred1, covs1 = simulate_kalman_filter(
            model, fps, track, raw_data.get_some_cams())
        loss = compute_loss(pred0, covs0, track, w_mse) \
            + compute_loss(pred1, covs1, track, w_mse)
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        optimizer.step()
        total_loss += loss.item()
        count += 1
    return total_loss / count


def eval_epoch(model, loader, raw_data, w_mse: float) -> float:
    """
    Run a single evaluation round over the given loader. This is intended to be
    used after each epoch to evaluate the performance on the validation set.
    """
    model.eval()
    total_loss = 0
    count = 0
    # Disable gradients to save memory and compute.
    with torch.inference_mode():
        for fps, track in loader:
            fps = fps.to(model.device, non_blocking=True)
            track = track.to(model.device, non_blocking=True)
            pred0, covs0, pred1, covs1 = simulate_kalman_filter(
                model, fps, track, raw_data.get_some_cams())
            loss = compute_loss(pred0, covs0, track, w_mse) \
                + compute_loss(pred1, covs1, track, w_mse)
            total_loss += loss.item()
            count += 1
    return total_loss / count


def train_epochs(nets: NetStorage, train, val, raw_data, num_epochs: int, w_mse: float, callback=None):
    """
    Perform multiple training epochs, recoding the history of both training
    and validation loss in the given log directory. This will train using the
    saved optimizer, model, and learning rate schedule from the provided net
    storage.
    """
    # Lower precision to benefit from certain hardware support.
    torch.set_float32_matmul_precision('high')
    model = nets.net
    optimizer = nets.optimizer
    scheduler = nets.scheduler
    for epoch in range(nets.step + 1, num_epochs):
        tr_loss = train_epoch(model, train, raw_data, optimizer, w_mse)
        val_loss = eval_epoch(model, val, raw_data, w_mse)
        nets.save_network(epoch, {
            "tr_loss": tr_loss, "val_loss": val_loss,
        }, model, optimizer, scheduler)
        scheduler.step(val_loss)
        print(f"epoch {epoch}; train: loss {tr_loss}; val: loss {val_loss}")
        if callback is not None and callback(val_loss, epoch):
            print("stopping via callback")
            break
    else:
        print("max epoch reached")


def train_epochs_in(num_epochs: int, nets_dir: str | None, stat_dir: str | None, model, w_mse=1.0, testing=False, callback=None):
    """
    Perform multiple training epochs. This will initialize the net storage in
    case we are starting a fresh run, and resume the existing run otherwise. The
    optimizer and learning rate schedule will be initializer according to the
    passed configuration.
    """
    util.set_seed(42)
    nets = util.net_storage_in(nets_dir, stat_dir, model.to(util.DEVICE))
    raw_data = dataset.CmuPanopticDataset(
        f"{os.path.dirname(__file__)}/data/panoptic")
    full_train = dataset.KalmanDataset(
        f"{os.path.dirname(__file__)}/data/kalman/train")
    if testing:
        # This configuration is only for the sanity check, it is not used for the
        # actual training of the models.
        train = torch.utils.data.Subset(full_train, range(32, 128))
        val = torch.utils.data.Subset(full_train, range(32))
    else:
        train = full_train
        val = dataset.KalmanDataset(
            f"{os.path.dirname(__file__)}/data/kalman/val")
    train_loader = DataLoader(
        train, 32, shuffle=True, drop_last=True, num_workers=8,
        persistent_workers=True, pin_memory=True, prefetch_factor=4)
    val_loader = DataLoader(
        val, 32, shuffle=True, drop_last=True, num_workers=8,
        persistent_workers=True, pin_memory=True, prefetch_factor=4)
    train_epochs(nets, train_loader, val_loader,
                 raw_data, num_epochs, w_mse, callback)
