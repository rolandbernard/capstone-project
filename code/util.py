
import os
import json
import random

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import scipy.optimize

# Slightly lower precision for performance.
torch.set_float32_matmul_precision('high')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# The set of links between keypoints that make up the skeleton in the COCO pose model.
SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11],
    [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2],
    [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
]
# Links that should have a constant physical length over time.
RIGID_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11],
    [6, 12], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [1, 2],
    [0, 1], [0, 2], [1, 3], [2, 4], [3, 4], [4, 17], [3, 17]
]
RIGID_SKELETON_SYM = [
    0, 1, 0, 1, 2, 3, 3, 4, 5, 6, 5, 6, 7, 8, 8, 9, 9, 10, 11, 11
]
COLORS = [
    "#FF5733", "#33FF57", "#3357FF", "#FF33A8",
    "#33FFF5", "#F5FF33", "#A833FF", "#F5A8F5"
]
COLORS_TUPLE = [
    (255, 87, 51), (51, 255, 87), (51, 87, 255), (255, 51, 168),
    (51, 255, 245), (245, 255, 51), (168, 51, 255), (245, 168, 245)
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
    """ Remove all indices in `idx` from the input tensor at dimension `dim`. """
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


def net_storage_in(nets_dir: str | None, stat_dir: str | None, model, lr, compile=True):
    """
    Initialize or load the net storage from the specified directories. This
    function will take the necessary parameters from the supplied configuration.
    """
    optimizer = optim.Adam(model.train_parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    return NetStorage(nets_dir, stat_dir, model, optimizer, scheduler, compile)


def count_model_params(model):
    """
    This is a simple function that just computes the number of trainable
    parameters of the given model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_list(something) -> list:
    if isinstance(something, np.ndarray) or isinstance(something, torch.Tensor):
        return something.tolist()
    if isinstance(something, list):
        return [to_list(s) for s in something]
    return something


def save_json(file: str, data):
    """ Save a JSON file and create the directory if necessary. """
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, "w") as f:
        json.dump(data, f)


def load_json(file: str):
    """ Load some JSON file into a python object and return it. """
    with open(file, "r") as f:
        return json.load(f)


def save_tracks(
    file: str, cams: list, frames: list[list], fps: float,
    center: tuple[float, float, float] = (0, 0, 0), up: tuple[float, float, float] = (0, -1, 0)
):
    """ Save recorded tracking data to the given file. """
    save_json(file, {
        "cameras": [{k: to_list(c[k]) for k in ["R", "t", "K", "distCoef"]} for c in cams],
        "frames": [[{k: to_list(t[k]) for k in ["id", "kpts"]} for t in f] for f in frames],
        "fps": fps, "center": list(center), "up": list(up)
    })


def load_tracks(file: str):
    """ Load recorded tracking data from the given file. """
    data = load_json(file)
    return data["cameras"], data["frames"], data["fps"], data["center"], data["up"]


def evaluate_mot_metrics(gt_frames, pred_frames, dist_threshold=15.0):
    """ Computes MPJPE, MOTA, MOTP, Precision, and Recall from data. """
    total_gt_kpts, total_pred_kpts = 0, 0
    total_tp, total_fp, total_fn = 0, 0, 0
    total_id_switches = 0
    mpjpe_errors, mpjpe_count = 0, 0
    motp_errors = 0
    prev_gt_to_pred_map = {}
    # Only go until the minimum of gt and prediction since some videos are cut
    # short before ground truth values end.
    for idx in range(min(len(gt_frames), len(pred_frames))):
        frame_gt = gt_frames[idx] if idx < len(gt_frames) else {}
        frame_pred = pred_frames[idx] if idx < len(pred_frames) else {}
        total_gt_kpts += len(frame_gt)
        total_pred_kpts += len(frame_pred)
        # Build cost matrix based on MPJPE distance.
        cost_matrix = np.zeros((len(frame_gt), len(frame_pred)))
        for i, gt in enumerate(frame_gt):
            for j, pred in enumerate(frame_pred):
                gt_joints = np.array(gt["kpts"])
                valid = np.sum(np.abs(gt_joints), axis=1) != 0.0
                pred_joints = np.array(pred["kpts"])
                cost_matrix[i, j] = np.mean(
                    np.linalg.norm(gt_joints - pred_joints, axis=1),
                    where=valid
                )
        gt_ind, pred_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
        current_gt_to_pred_map = prev_gt_to_pred_map.copy()
        matches = 0
        for g, p in zip(gt_ind, pred_ind):
            mpjpe_errors += cost_matrix[g, p].item()
            mpjpe_count += 1
            # Apply gating threshold to only allow those with mean below threshold.
            if cost_matrix[g, p] <= dist_threshold:
                gt_id, pred_id = frame_gt[g]["id"], frame_pred[p]["id"]
                current_gt_to_pred_map[gt_id] = pred_id
                motp_errors += cost_matrix[g, p].item()
                matches += 1
                # Check for identity switch
                if gt_id in prev_gt_to_pred_map and prev_gt_to_pred_map[gt_id] != pred_id:
                    total_id_switches += 1
        total_tp += matches
        # Unmatched ground truth items are false negatives.
        total_fn += len(frame_gt) - matches
        # Unmatched predictions items are false positives.
        total_fp += len(frame_pred) - matches
        # Save assignment map for next iteration.
        prev_gt_to_pred_map = current_gt_to_pred_map
    # Compute metrics.
    mota = 1.0 - ((total_fn + total_fp + total_id_switches) / total_gt_kpts) \
        if total_gt_kpts > 0 else 0.0
    motp = motp_errors / total_tp if total_tp > 0 else 0.0
    mpjpe = mpjpe_errors / mpjpe_count if mpjpe_count > 0 else 0.0
    precision = total_tp / (total_tp + total_fp) \
        if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) \
        if (total_tp + total_fn) > 0 else 0.0
    # Output results as dictionary.
    return {
        "MPJPE": mpjpe,
        "MOTA": mota,
        "MOTP (Avg Miss Distance)": motp,
        "Precision": precision,
        "Recall": recall,
        "Counts": {
            "GT Objects": total_gt_kpts,
            "Pred Objects": total_pred_kpts,
            "True Positives": total_tp,
            "False Positives": total_fp,
            "False Negatives": total_fn,
            "ID Switches": total_id_switches
        }
    }


def evaluate_from_files(gt_file: str, pred_file: str, dist_threshold=15.0):
    """ Load results and ground truth from the given files and compute metrics. """
    _, gt_frames, _, _, _ = load_tracks(gt_file)
    _, pred_frames, _, _, _ = load_tracks(pred_file)
    return evaluate_mot_metrics(gt_frames, pred_frames, dist_threshold)


def per_point_cov(covar: torch.Tensor, num_dim: int = 3) -> torch.Tensor:
    """ Extract the block diagonal part of the given covariance matrix. """
    *Bs, N, N = covar.shape
    by_point = covar.view(-1, N // num_dim, num_dim, N // num_dim, num_dim)
    blocks = torch.diagonal(by_point, dim1=1, dim2=3)
    return blocks.permute(0, 3, 1, 2).view(*Bs, -1, num_dim, num_dim)


def sanitize_covariance(cov: torch.Tensor, floor: float = 1e-6) -> torch.Tensor:
    """ Forces symmetry and projects back onto the positive-definite cone. """
    sym_cov = 0.5 * (cov + cov.mT)
    eigs, vecs = torch.linalg.eigh(sym_cov)
    eigs = torch.clamp(eigs, min=floor)
    return vecs @ torch.diag_embed(eigs) @ vecs.mT


def diagnose_covariance(cov: torch.Tensor, name: str = "Covariance Matrix"):
    """
    Analyzes a large covariance matrix for symmetry, positive-definiteness, 
    conditioning, and extreme variance disparities.
    """
    results = {}
    print(f"\n================ {name} ({cov.shape}) ================")
    sym_err = torch.max(torch.abs(cov - cov.mT)).item()
    results['max_asymmetry'] = sym_err
    print(f"Max Asymmetry Error | P - P^T | : {sym_err:.2e}")
    if sym_err > 1:
        print("  CRITICAL: Matrix has lost symmetry")
    elif sym_err > 1e-3:
        print("  WARNING: Matrix has lost symmetry")
    has_nan = torch.isnan(cov).any().item()
    has_inf = torch.isinf(cov).any().item()
    if has_nan or has_inf:
        print(
            f"  CRITICAL: Matrix contains NaN ({has_nan}) or Inf ({has_inf})")
    else:
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
        if min_eig > 0:
            cond_num = max_eig / min_eig
            results['condition_number'] = cond_num
            print(f"Condition Number (λmax/λmin) : {cond_num:.2e}")
            if cond_num > 1e10:
                print("  WARNING: Matrix is severely ill-conditioned")
        else:
            print("  CRITICAL: Matrix is NOT Positive-Definite")
        diag = torch.diagonal(cov, dim1=-2, dim2=-1)
        neg_diag_indices = (diag <= 0).nonzero(as_tuple=True)[0].tolist()
        if neg_diag_indices:
            print(
                f"  State indices with non-positive variance: {neg_diag_indices}")
        print(
            f"Variance Range (Min/Max Diag): {diag.min().item():.2e} / {diag.max().item():.2e}")


def check_covariance(cov: torch.Tensor, label: str | None = None):
    """ Check that the given covariance matrix is SPD and fail otherwise. """
    try:
        torch.linalg.cholesky(cov)
    except:
        if label is not None:
            print(label)
        diagnose_covariance(cov)
        torch.linalg.cholesky(cov)
