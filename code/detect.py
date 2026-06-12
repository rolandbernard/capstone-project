
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ultralytics import YOLO
from ultralytics.nn.tasks import PoseModel
from ultralytics.nn.modules import Detect, Pose26, Conv
import scipy.optimize

import util
import dataset
from camera import Camera
from util import NetStorage


class PoseDetector:
    """
    This is an wrapper class around the YOLO based pose estimation models. Given
    an image it produces as output a tensor of shape (P, K*3) where P is the
    number of detected persons an K = 17 is the number of keypoints. It also
    gives an estimated covariance matrix based on keypoint visibility.
    """

    def __init__(
        self, model_name: str = "yolo26n-pose", threshold: float = 0.5, kpt_threshold: float = 0.25,
        min_keypoint: int = 3, var_min: float = 16.0, var_vis: float = 0.005, var_inv: float = 5.0,
        path: str = "./nets", compile: bool = True
    ):
        model: PoseModel = YOLO(
            f"{path}/{model_name}.pt").model  # type: ignore
        model.eval()
        if compile:
            model.compile()
        self.model = model
        self.threshold = threshold
        self.kpt_threshold = kpt_threshold
        self.min_keypoint = min_keypoint
        self.var_min = var_min
        self.var_vis = var_vis
        self.var_inv = var_inv
        self.num_keypoint = 17

    def detect_base(self, images: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Basic version of the detection loop that expects the images to be already
        batched in the expected format.
        """
        pred, _ = self.model(images)
        results = []
        for img_res in pred:
            valid = img_res[(img_res[:, 4] > self.threshold) &
                            ((img_res[:, 8::3] > self.kpt_threshold).sum() >= self.min_keypoint)]
            valid_points = valid[:, 6:].view(-1, self.num_keypoint, 3)
            # Compute variance based on bounding box size and kpt visibility.
            bb_size = torch.linalg.vector_norm(
                valid[:, 2:4] - valid[:, 0:2], dim=1, keepdim=True)
            vis = torch.clamp(
                (valid_points[:, :, 2] - self.kpt_threshold) / (1.0 - self.kpt_threshold), min=0)
            var = self.var_min + bb_size * bb_size * \
                (self.var_vis / (vis + self.var_vis / self.var_inv))
            results.append((
                valid_points[:, :, 0:2].reshape(-1, self.num_keypoint*2),
                torch.kron(torch.diag_embed(var),
                           torch.eye(2, device=var.device))
            ))
        return results

    def detect_simple(self, images: torch.Tensor | list[torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Run the detection algorithm and return discovered keypoints. We expect
        a batch of images and return a batch of results. For the results we return
        for P persons two tensors of shape (P, 17*2) for the positions of all 17
        keypoints and and a tensor of shape (P, 17*2, 17*2) for the covariance.
        """
        with torch.inference_mode():
            if isinstance(images, list):
                images = torch.stack(images)
            if images.shape[-1] == 3:
                images = images.permute(0, 3, 1, 2)
            if images.dtype == torch.uint8:
                images = images.to(torch.float32) / 255.0
            return self.detect_base(images)

    def detect(self, cams: list[Camera], images: torch.Tensor | list[torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        This is like `detect_simple`, but the returned keypoints are in normalized
        camera coordinates rather than in pixel space, effectively removing the
        camera intrinsics.
        """
        return [
            (cam.undistort_points(pos), cam.undistort_covars(cov))
            for cam, (pos, cov) in zip(cams, self.detect_simple(images))
        ]

    def to(self, *args, **kargs):
        """
        Apply the PyTorch `.to` method to the contained model.
        """
        self.model = self.model.to(*args, **kargs)
        return self


class CustomPoseDetector(PoseDetector):
    """
    This is an wrapper class around the YOLO model with the custom keypoint
    prediction head. This extends `PoseDetector` even though it does not share
    nearly any of its functionality.
    """

    def __init__(self, model: CustomHeadedYolo, threshold: float = 0.5, compile: bool = True):
        model.eval()
        if compile:
            model.compile()
        self.model = model
        self.threshold = threshold
        self.num_keypoint = 17

    def detect_base(self, images: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Run the detection algorithm and return discovered keypoints.
        """
        pred = self.model(images)
        valid_mask = pred["one2one"]["scores"] > self.threshold
        results = []
        for b in range(images.shape[0]):
            valid_idx = valid_mask[b].flatten().nonzero(as_tuple=True)[0]
            kpts = pred["kpts_extra"][b, :, valid_idx] \
                .view(self.num_keypoint, 5, -1).permute(2, 0, 1)
            mu = kpts[..., :2]
            a, b, c = kpts[..., 2], kpts[..., 3], kpts[..., 4]
            # Compute variance based on cholesky factors.
            cov = torch.stack([
                torch.stack([a*a, a*c], dim=-1),
                torch.stack([a*c, c*c + b*b], dim=-1)
            ], dim=-2)
            # Diagonalize the covariances assuming independence.
            cov = torch.diag_embed(cov.permute(0, 2, 3, 1), dim1=1, dim2=3)
            results.append((
                mu.reshape(-1, self.num_keypoint*2),
                cov.reshape(-1, self.num_keypoint*2, self.num_keypoint*2)
            ))
        return results


class CustomPose(Pose26):
    """
    A custom head for the YOLO26 model that saves also the features of the pose
    head so they can be reused later to train to natively output mean and covariance
    for each keypoint position instead of visibility value. The original head is kept.
    """

    def __init__(self, nc: int = 1, kpt_shape: tuple = (17, 3), reg_max=1, end2end=True, ch: tuple = (64, 128, 256)):
        super().__init__(nc, kpt_shape, reg_max, end2end, ch)

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: nn.Module,
        cls_head: nn.Module,
        pose_head: nn.ModuleList,
        kpts_head: nn.ModuleList,
        kpts_sigma_head: nn.ModuleList,
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes, class probabilities, and keypoints."""
        preds = Detect.forward_head(self, x, box_head, cls_head)
        if pose_head is not None:
            bs = x[0].shape[0]
            features = [pose_head[i](x[i]) for i in range(self.nl)]
            preds["kpts_features"] = features  # type: ignore
            preds["kpts"] = torch.cat([
                kpts_head[i](features[i]).view(bs, self.nk, -1)
                for i in range(self.nl)
            ], dim=2)
            if self.training:
                preds["kpts_sigma"] = torch.cat([
                    kpts_sigma_head[i](features[i]).view(bs, self.nk_sigma, -1)
                    for i in range(self.nl)
                ], dim=2)
        return preds


class CustomHeadedYolo(nn.Module):
    """
    Small wrapper to install the custom pose head at the end of a standard YOLO26
    human pose estimation model.
    """

    def __init__(self, original_model: str | PoseModel = "yolo26n-pose", path: str = "./nets"):
        super().__init__()
        # Acquire the base model.
        if isinstance(original_model, str):
            original_model = YOLO(
                f"{path}/{original_model}.pt").model  # type: ignore
        self.base_net: PoseModel = original_model  # type: ignore
        # Freeze the base model.
        self.base_net.eval()
        for param in self.base_net.parameters():
            param.requires_grad = False
        # Create the extra head.
        self.extra_head = nn.ModuleList(nn.Sequential(
            Conv(ch, 100, 3),
            Conv(100, 100, 3),
            Conv(100, 100, 3),
            nn.Conv2d(100, 17*5, 1)
        ) for ch in (64, 128, 256))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def anchors(self):
        return self.base_net.model[-1].anchors

    @property
    def strides(self):
        return self.base_net.model[-1].strides

    def train(self, train=True):
        super().train(train)
        self.base_net.eval()
        return self

    def extra_parameters(self):
        """
        Get the extra parameters that have been added by the custom head. During
        training we will be freezing all other parameters.
        """
        return self.extra_head.parameters()

    def forward(self, x):
        bs = x.shape[0]
        with torch.no_grad():
            assert not self.base_net.training
            _, pred = self.base_net(x)
        extra = torch.cat([
            self.extra_head[i](features).view(bs, 17*5, -1)
            for i, features in enumerate(pred["one2one"]["feats"])
        ], dim=2).view(bs, 17, 5, -1)
        pred["kpts_extra"] = torch.concat([
            (extra[:, :, 0:2] + self.anchors) * self.strides,
            torch.exp(
                torch.clamp(extra[:, :, 2:4], min=-4, max=6)
            ) * self.strides,
            extra[:, :, 4:5] * self.strides,
        ], dim=2).view(bs, 17*5, -1)
        return pred


def compute_nll(pred: torch.Tensor, gt: torch.Tensor, eps=1e-5) -> torch.Tensor:
    """
    Computes the weighted negative log likelihood loss for 2D Gaussian in the
    predictions against the ground truth.
    """
    mu, a, b, c = pred[..., :2], pred[..., 2], pred[..., 3], pred[..., 4]
    gt_xy, w = gt[..., :2], gt[..., 2]
    L = torch.stack([
        torch.stack([a, torch.zeros_like(a)], dim=-1),
        torch.stack([c, b], dim=-1)
    ], dim=-2)
    logdet = 2.0 * (torch.log(a + eps) + torch.log(b + eps))
    diff = (gt_xy - mu).unsqueeze(-1)
    v = torch.linalg.solve_triangular(L, diff, upper=False)
    mahalanobis = v.mT @ v
    nll = logdet + mahalanobis
    weighted_nll = nll * w
    return torch.mean(weighted_nll) + 0.1 * torch.mean(diff*diff)


def compute_loss(pred, gt: torch.Tensor, model, threshold: float = 0.0, eps=1e-5):
    """
    Compute the loss between the predictions and the ground truth. First, match
    the detections against the known ground truth and then compute the negative
    log likelihood over the resulting matches.
    """
    bs = gt.shape[0]
    anchors, strides = model.anchors, model.strides
    valid_mask = pred["one2one"]["scores"].detach() > threshold
    all_kpts = pred["one2one"]["kpts"].detach().clone()
    all_kpts = all_kpts.view(bs, 17, 3, -1)
    all_kpts[:, :, :2] = (all_kpts[:, :, :2] + anchors) * strides
    preds = []
    gts = []
    for b in range(bs):
        valid_idx = valid_mask[b].flatten().nonzero(as_tuple=True)[0]
        if len(valid_idx) == 0:
            continue
        b_kpts = all_kpts[b, :, :, valid_idx].permute(2, 0, 1)
        b_gt = gt[b, torch.sum(gt[b, :, :, 2], dim=-1) > 0.01]
        dist_matrix = b_kpts[:, :, :2].unsqueeze(1) \
            - b_gt[:, :, :2].unsqueeze(0)
        dist_matrix *= dist_matrix
        weight = torch.sigmoid(b_kpts[:, :, 2]).unsqueeze(1) \
            * b_gt[:, :, 2].unsqueeze(0)
        cost_matrix = torch.sum(torch.sum(dist_matrix, dim=-1) * weight, dim=-1) \
            / (torch.sum(weight, dim=-1) + eps)
        # 6. Hungarian Matching (Push to CPU only for the solver)
        cost_np = cost_matrix.cpu().numpy()
        row_idx, col_idx = scipy.optimize.linear_sum_assignment(cost_np)
        matched_pred_indices = valid_idx[row_idx]
        preds.append(
            pred["kpts_extra"][b, :, matched_pred_indices].view(17, 5, -1).permute(2, 0, 1))
        gts.append(b_gt[col_idx])
    if len(preds) == 0:
        return torch.tensor(0.0, device=gt.device, requires_grad=True)
    return compute_nll(torch.concat(preds, dim=0), torch.concat(gts, dim=0), eps)


def train_epoch(model, loader, optimizer):
    """
    Perform a single training epoch. Also computes the average training loss
    over the course of the epoch.
    """
    # Ensure we are in training mode.
    model.train()
    total_loss = 0
    count = 0
    for img, gts in loader:
        img = img.to(model.device, non_blocking=True)
        gts = gts.to(model.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(img)
        loss = compute_loss(pred, gts, model)
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        optimizer.step()
        total_loss += loss.item()
        count += 1
    return total_loss / count


def eval_epoch(model, loader):
    """
    Run a single evaluation round over the given loader. This is intended to be
    used after each epoch to evaluate the performance on the validation set.
    """
    model.eval()
    total_loss = 0
    count = 0
    # Disable gradients to save memory and compute.
    with torch.inference_mode():
        for img, gts in loader:
            img = img.to(model.device, non_blocking=True)
            gts = gts.to(model.device, non_blocking=True)
            pred = model(img)
            loss = compute_loss(pred, gts, model)
            total_loss += loss.item()
            count += 1
    return total_loss / count


def train_epochs(nets: NetStorage, train, val, num_epochs: int, callback=None):
    """
    Perform a multiple training epochs, recoding the history of both training
    and validation loss in the given log directory. This will train using the
    saved optimizer, model, and learning rate schedule from the provided net
    storage. Uses pixel-wise cross-entropy as the loss.
    """
    # Lower precision to benefit from certain hardware support.
    torch.set_float32_matmul_precision('high')
    model = nets.net
    optimizer = nets.optimizer
    scheduler = nets.scheduler
    for epoch in range(nets.step + 1, num_epochs):
        tr_loss = train_epoch(model, train, optimizer)
        val_loss = eval_epoch(model, val)
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


def train_epochs_in(num_epochs: int, nets_dir: str | None, stat_dir: str | None, model, testing=False, callback=None):
    """
    Perform a multiple training epochs. This will initialize the net storage in
    case we are starting a fresh run, and resume the existing run otherwise. The
    optimizer and learning rate schedule will be initializer according to the
    passed configuration. The test and train split are generated.
    """
    util.set_seed(42)
    nets = util.net_storage_in(nets_dir, stat_dir, model.to(util.DEVICE))
    full_train = dataset.YoloDataset(
        f"{os.path.dirname(__file__)}/data/yolo/train")
    if testing:
        # This configuration is only for the sanity check, it is not used for the
        # actual training of the models.
        train = torch.utils.data.Subset(full_train, range(32))
        val = train
    else:
        train = full_train
        val = dataset.YoloDataset(f"{os.path.dirname(__file__)}/data/yolo/val")
    train_loader = DataLoader(
        train, 32, shuffle=True, drop_last=True, num_workers=8,
        persistent_workers=True, pin_memory=True, prefetch_factor=4)
    val_loader = DataLoader(
        val, 32, shuffle=True, drop_last=True, num_workers=8,
        persistent_workers=True, pin_memory=True, prefetch_factor=4)
    train_epochs(nets, train_loader, val_loader, num_epochs, callback)
