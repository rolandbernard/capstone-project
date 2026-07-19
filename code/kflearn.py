
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import util
import dataset
import kalman
from camera import Camera
from util import NetStorage


def diagnose_covariance(cov: torch.Tensor, name: str = "Covariance Matrix") -> dict:
    """
    Analyzes a large covariance matrix for symmetry, positive-definiteness, 
    conditioning, and extreme variance disparities.
    """
    results = {}
    print(f"\n================ {name} ({cov.shape}) ================")

    # 1. Symmetry Check
    asym_err = torch.max(torch.abs(cov - cov.mT)).item()
    results['max_asymmetry'] = asym_err
    print(f"Max Asymmetry Error | P - P^T | : {asym_err:.2e}")
    if asym_err > 1e-4:
        print("  ⚠️ WARNING: Matrix has lost symmetry.")

    # 2. NaN / Inf Check
    has_nan = torch.isnan(cov).any().item()
    has_inf = torch.isinf(cov).any().item()
    if has_nan or has_inf:
        print(
            f"  ❌ CRITICAL: Matrix contains NaN ({has_nan}) or Inf ({has_inf})")
        return results

    # 3. Eigenvalue Spectrum & Positive-Definiteness
    # eigh is optimized for symmetric matrices
    sym_cov = 0.5 * (cov + cov.mT)
    eigenvalues = torch.linalg.eigvalsh(sym_cov)

    min_eig = eigenvalues.min().item()
    max_eig = eigenvalues.max().item()
    num_neg = (eigenvalues < 0).sum().item()

    results['min_eig'] = min_eig
    results['max_eig'] = max_eig
    results['num_neg_eigs'] = num_neg

    print(f"Min Eigenvalue            : {min_eig:.2e}")
    print(f"Max Eigenvalue            : {max_eig:.2e}")
    print(f"Negative Eigenvalue Count : {num_neg} / {cov.shape[-1]}")

    # 4. Condition Number (Dynamic Range Ratio)
    if min_eig > 0:
        cond_num = max_eig / min_eig
        results['condition_number'] = cond_num
        print(f"Condition Number (λmax/λmin) : {cond_num:.2e}")
        if cond_num > 1e10:
            print(
                "  ⚠️ WARNING: Matrix is severely ill-conditioned (loss of precision likely).")
    else:
        print("  ❌ CRITICAL: Matrix is NOT Positive-Definite!")

    # 5. Diagonal Variance Analysis
    diag = torch.diagonal(cov, dim1=-2, dim2=-1)
    neg_diag_indices = (diag <= 0).nonzero(as_tuple=True)[0].tolist()

    if neg_diag_indices:
        print(
            f"  ❌ State indices with non-positive variance: {neg_diag_indices}")

    # Range of variances across the 175 states
    print(
        f"Variance Range (Min/Max Diag): {diag.min().item():.2e} / {diag.max().item():.2e}")

    return results


def simulate_kalman_filter(
    model: kalman.LearnedPhysics, fps: torch.Tensor, track: torch.Tensor,
    cams: list[Camera], v_init=20.0, v_vis_min=5.0, v_vis_max=50.0, v_inv=1000.0
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
        track[..., 0, :, :].view(*Bs, -1)
        + torch.randn(*Bs, K*D, device=track.device) * v_init,
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
        ncov = [
            torch.where(
                ((pts[..., 0] > 0) & (pts[..., 0] < 640)
                 & (pts[..., 1] > 0) & (pts[..., 1] < 480)).unsqueeze(-1),
                v_vis_min + torch.rand_like(pts) * v_vis_max,
                torch.full_like(pts, v_inv)
            )
            for pts in proj]
        ob_fs = [lambda x, cam=cam: cam.project_pinhole(x[:K*D].view(-1, D)).flatten()
                 for cam in cams]
        min_bounds = torch.tensor([0.0, 0.0], device=means.device)
        max_bounds = torch.tensor([640.0, 480.0], device=means.device)
        ob_ms = [
            cam.undistort_points(
                (pts + torch.randn_like(pts) * cov)
                .clamp(min=min_bounds, max=max_bounds).view(*Bs, -1))
            for cam, pts, cov in zip(cams, proj, ncov)]
        ob_vs = [
            cam.undistort_covars(torch.diag_embed(cov.view(*Bs, -1)))
            for cam, cov in zip(cams, ncov)]
        ob_fs.append(lambda x: model.pseudo_obs(x))  # type: ignore
        ob_ms.append(model.constr_val.expand(*Bs, *model.constr_val.shape))
        ob_vs.append(model.constr_cov.expand(*Bs, *model.constr_cov.shape))
        ob_f, ob_m, ob_v = kalman.emerge_obs(ob_fs, ob_ms, ob_vs)
        print("update")
        means, covs = kalman.eupdate(means, covs, ob_m, ob_v, ob_f)
        for cov in covs.view(-1, *covs.shape[-2:]):
            try:
                torch.linalg.cholesky(cov)
            except:
                print("! covs")
                diagnose_covariance(cov)
                return
        # Save post-update prediction.
        pred1.append(means[..., :K*D])
        covs1.append(covs[..., :K*D, :K*D])
        # Predict next state. (Only if not the last state.)
        if t != T - 1:
            print("predict")
            means, covs = model.predict_train(dt, means, covs)
            for cov in covs.view(-1, *covs.shape[-2:]):
                try:
                    torch.linalg.cholesky(cov)
                except:
                    print("! covs")
                    diagnose_covariance(cov)
                    return
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


def train_epoch(model, loader, raw_data, optimizer, w_mse: float) -> float:
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
        nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        optimizer.step()
        total_loss += loss.item()
        count += 1
        print("iter")
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
        train, 1, shuffle=not testing, drop_last=True, num_workers=8,
        persistent_workers=True, pin_memory=True, prefetch_factor=4)
    val_loader = DataLoader(
        val, 1, shuffle=not testing, drop_last=True, num_workers=8,
        persistent_workers=True, pin_memory=True, prefetch_factor=4)
    train_epochs(nets, train_loader, val_loader,
                 raw_data, num_epochs, w_mse, callback)
