
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import PoseModel
from ultralytics.nn.modules import Detect, Pose26, Conv

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


class CustomPose(Pose26):
    """
    A custom head for the YOLO26 model that is trained to natively output mean
    and covariance for each keypoint position instead of visibility value. The
    original head is kept.
    """

    def __init__(self, nc: int = 1, kpt_shape: tuple = (17, 3), reg_max=16, end2end=True, ch: tuple = ()):
        super().__init__(nc, kpt_shape, reg_max, end2end, ch)
        c4 = max(ch[0] // 4, kpt_shape[0] * (kpt_shape[1] + 2))
        self.nk_v2 = kpt_shape[0]*5
        self.kpts_v2 = nn.ModuleList(
            nn.Sequential(Conv(c4, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk_v2, 1)) for _ in ch)

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
            bs = x[0].shape[0]  # batch size
            features = [pose_head[i](x[i]) for i in range(self.nl)]
            preds["kpts"] = torch.cat([
                kpts_head[i](features[i]).view(bs, self.nk, -1)
                for i in range(self.nl)
            ], dim=2)
            preds["kpts_v2"] = torch.cat([
                self.kpts_v2[i](features[i]).view(bs, self.nk_v2, -1)
                for i in range(self.nl)
            ], dim=2)
        return preds

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities, concatenated with keypoints."""
        preds = super()._inference(x)
        return torch.cat([preds, self.kpts_v2_decode(x["kpts_v2"])], dim=1)

    def kpts_v2_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        """Decode keypoints from predictions."""
        bs = kpts.shape[0]
        y = kpts.view(bs, self.kpt_shape[0], 5, -1)
        return torch.stack([
            (y[:, :, 0] + self.anchors[0]) * self.strides,
            (y[:, :, 1] + self.anchors[1]) * self.strides,
            y[:, :, 2] * self.strides,
            y[:, :, 3] * self.strides,
            y[:, :, 4] * self.strides,
        ], dim=2).view(bs, self.nk_v2, -1)


class CustomHeadedYolo(nn.Module):
    """
    Small wrapper to install the custom pose head at the end of a standard YOLO26
    human pose estimation model.
    """

    def __init__(self, original_model: str | PoseModel = "yolo26n-pose", path: str = "./nets"):
        super().__init__()
        if isinstance(original_model, str):
            original_model = YOLO(
                f"{path}/{model_name}.pt").model  # type: ignore
        self.base_net: PoseModel = original_model  # type: ignore
        old_head = self.base_net.model[-1]
        old_state_dict = old_head.state_dict()
        self.extra_head = CustomPose()
        self.extra_head.load_state_dict(old_state_dict, strict=False)
        self.base_net.model[-1] = self.extra_head

    def extra_parameters(self):
        """
        Get the extra parameters that have been added by the custom head. During
        training we will be freezing all other parameters.
        """
        return self.extra_head.kpts_v2.parameters()

    def forward(self, x):
        return self.base_net(x)


def compute_loss(pred: torch.Tensor, gt: torch.Tensor, criterion):
    raise NotImplementedError


def train_epoch(model, loader, optimizer, criterion, scaler):
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
        with torch.autocast(model.device.type):
            pred, _ = model(img)
            loss = compute_loss(pred, gts, criterion)
        scaler.scale(loss).backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        count += 1
    return total_loss / count


def eval_epoch(model, loader, criterion):
    """
    Run a single evaluation round over the given loader. This is intended to be
    used after each epoch to evaluate the performance on the validation set.
    """
    model.eval()
    total_loss = 0
    count = 0
    # Disable gradients to save memory and compute.
    with torch.inference_mode():
        for x, y, ts, ts_y in loader:
            x = x.to(model.device, non_blocking=True)
            y = y.to(model.device, non_blocking=True)
            ts = ts.to(model.device, non_blocking=True)
            ts_y = ts_y.to(model.device, non_blocking=True)
            with torch.autocast(model.device.type):
                logits = model(x, ts, ts_y)
                loss = compute_loss(logits, y, criterion)
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
    scaler = torch.GradScaler(device=model.device.type)
    criterion = nn.MSELoss()
    for epoch in range(nets.step + 1, num_epochs):
        tr_loss = train_epoch(model, train, optimizer, criterion, scaler)
        val_loss = eval_epoch(model, val, criterion)
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
