
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import util
import dataset
import kalman
from camera import Camera
from util import NetStorage


def simulate_kalman_filter(
    model: kalman.LearnedPhysics, fps: torch.Tensor, track: torch.Tensor,
    cams: list[Camera], v_vis_min=4.0, v_vis_max=50.0, v_inv=1e3, checks=False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform a simulated run of the Kalman filter on the given tracks. Observations
    are simulated by applying known normal noise to the true projections. The noise
    level itself is randomly generated. Large noise is added for invisible points.
    Predictions before and after update are returned for each time step.
    """
    dt = 1.0 / fps
    *Bs, T, K, D = track.shape
    pred0, covs0, pred1, covs1 = [], [], [], []
    means = torch.concat([
        track[..., 0, :, :].view(*Bs, -1),
        model.init_mean[K*D:].expand(*Bs, -1)
    ], dim=-1)
    covs = model.init_cov.expand(*Bs, *model.init_cov.shape)
    for t in range(T):
        gt = track[..., t, :, :].contiguous()
        # Save pre-update prediction.
        pred0.append(means[..., :K*D])
        covs0.append(covs[..., :K*D, :K*D])
        # Generate fake observations and perform update.
        proj = [cam.project(gt) for cam in cams]
        nstd = [
            torch.where(
                ((pts[..., 0] > 0) & (pts[..., 0] < 640)
                 & (pts[..., 1] > 0) & (pts[..., 1] < 480)).unsqueeze(-1),
                v_vis_min + torch.rand_like(pts) * (v_vis_max - v_vis_min),
                torch.full_like(pts, v_inv)
            )
            for pts in proj]
        ob_fs = [lambda x, cam=cam: cam.project_pinhole(x[:K*D].view(-1, D)).flatten()
                 for cam in cams]
        min_bounds = torch.tensor([0.0, 0.0], device=means.device)
        max_bounds = torch.tensor([640.0, 480.0], device=means.device)
        ob_ms = [
            cam.undistort_points(
                (pts + torch.randn_like(pts) * std.clamp(max=v_vis_max))
                .clamp(min=min_bounds, max=max_bounds).view(*Bs, -1))
            for cam, pts, std in zip(cams, proj, nstd)]
        ob_vs = [
            cam.undistort_covars(torch.diag_embed((std * std).view(*Bs, -1)))
            for cam, std in zip(cams, nstd)]
        ob_fs.append(lambda x: model.pseudo_obs(x))  # type: ignore
        ob_ms.append(model.constr_val.expand(*Bs, *model.constr_val.shape))
        ob_vs.append(model.constr_cov.expand(*Bs, *model.constr_cov.shape))
        ob_f, ob_m, ob_v = kalman.emerge_obs(ob_fs, ob_ms, ob_vs)
        means, covs = kalman.eupdate(means, covs, ob_m, ob_v, ob_f)
        if checks:
            print("update")
            for i, cov in enumerate(covs.view(-1, *covs.shape[-2:])):
                print(i, end=" ")
                util.check_covariance(cov)
            print()
        # Save post-update prediction.
        pred1.append(means[..., :K*D])
        covs1.append(covs[..., :K*D, :K*D])
        # Predict next state. (Only if not the last state.)
        if t != T - 1:
            means, covs = model.predict_train(dt, means, covs)
            if checks:
                print("predict")
                for i, cov in enumerate(covs.view(-1, *covs.shape[-2:])):
                    print(i, end=" ")
                    util.check_covariance(cov)
                print()
    return torch.stack(pred0, dim=-2), torch.stack(covs0, dim=-3), \
        torch.stack(pred1, dim=-2), torch.stack(covs1, dim=-3)


def compute_loss(pred, covs, gt: torch.Tensor, w_mse: float) -> torch.Tensor:
    """
    Compute the loss between the predictions and the ground truth. Both are
    assumed to be sequences of 3d keypoint detections. The prediction is composed
    of both a mean and a covariance matrix. The loss is a combination a MSE and
    a NLL term.
    """
    *Bs, T, K, D = gt.shape
    gt_flat = gt.view(*Bs, T, K * D)
    mse_loss = nn.functional.mse_loss(pred, gt_flat)
    dist = torch.distributions.MultivariateNormal(pred, covs)
    nll_loss = -dist.log_prob(gt_flat).mean()
    return nll_loss + w_mse * mse_loss


def train_epoch(model: kalman.LearnedPhysics, loader, raw_data, optimizer, w_mse: float) -> float:
    """
    Perform a single training epoch. Also computes the average training loss
    over the course of the epoch.
    """
    # Ensure we are in training mode.
    model.train()
    total_loss = 0
    count = 0
    cams = [cam.to(model.device) for cam in raw_data.get_some_cams()]
    for fps, track in loader:
        fps = fps.to(model.device, non_blocking=True)
        track = track.to(model.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred0, covs0, pred1, covs1 \
            = simulate_kalman_filter(model, fps, track, cams)
        loss = compute_loss(pred0, covs0, track, w_mse) \
            + compute_loss(pred1, covs1, track, w_mse)
        loss.backward()
        nn.utils.clip_grad_value_(model.train_parameters(), clip_value=1.0)
        optimizer.step()
        total_loss += loss.item()
        count += 1
        print("==== iter ====")
        model.sanitize_covariances()
    return total_loss / count


def eval_epoch(model, loader, raw_data, w_mse: float) -> float:
    """
    Run a single evaluation round over the given loader. This is intended to be
    used after each epoch to evaluate the performance on the validation set.
    """
    model.eval()
    total_loss = 0
    count = 0
    cams = [cam.to(model.device) for cam in raw_data.get_some_cams()]
    # Disable gradients to save memory and compute.
    with torch.inference_mode():
        for fps, track in loader:
            fps = fps.to(model.device, non_blocking=True)
            track = track.to(model.device, non_blocking=True)
            pred0, covs0, pred1, covs1 \
                = simulate_kalman_filter(model, fps, track, cams)
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
        train, 32, shuffle=not testing, drop_last=True, num_workers=8,
        persistent_workers=True, pin_memory=True, prefetch_factor=4)
    val_loader = DataLoader(
        val, 32, shuffle=not testing, drop_last=True, num_workers=8,
        persistent_workers=True, pin_memory=True, prefetch_factor=4)
    train_epochs(nets, train_loader, val_loader,
                 raw_data, num_epochs, w_mse, callback)
