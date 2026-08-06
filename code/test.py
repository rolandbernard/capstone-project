
import sys

import torch
import numpy as np
from scipy.stats import chi2

import dataset
import visualize
import kflearn
import tracker
import kalman
import util
from camera import Camera


def simulate_kalman_filter(
    model: kalman.LearnedPhysics, fps: torch.Tensor, track: torch.Tensor,
    cams: list[Camera], v_vis_min=25.0, v_vis_max=25.0, v_inv=1e5, checks=False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dt = 1.0 / fps
    *Bs, T, K, D = track.shape
    pred0, covs0, pred1, covs1 = [], [], [], []
    means = model.init_mean.expand(*Bs, *model.init_mean.shape)
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
                v_vis_min, #+ torch.rand_like(pts) * (v_vis_max - v_vis_min),
                torch.full_like(pts, v_inv)
            )
            for pts in proj]
        ob_fs = [lambda x, cam=cam: cam.project_pinhole(x[:K*D].view(-1, D)).flatten()
                 for cam in cams]
        min_bounds = torch.tensor([0.0, 0.0], device=means.device)
        max_bounds = torch.tensor([640.0, 480.0], device=means.device)
        ob_ms = [
            ((cam.undistort_points(
                (pts )
                .clamp(min=min_bounds, max=max_bounds)
                )+ torch.randn_like(pts) * std.clamp(max=v_vis_max) / 750)

                .view(*Bs, -1))
            for cam, pts, std in zip(cams, proj, nstd)]
        ob_vs = [
                torch.diag_embed((std * std).view(*Bs, -1)) / (750*750) * 2
            for cam, std in zip(cams, nstd)]
        # ob_fs.append(lambda x: model.pseudo_obs(x))  # type: ignore
        # ob_ms.append(model.constr_val.expand(*Bs, *model.constr_val.shape))
        # ob_vs.append(model.constr_cov.expand(*Bs, *model.constr_cov.shape))
        ob_f, ob_m, ob_v = kalman.emerge_obs(ob_fs, ob_ms, ob_vs)
        means, covs = kalman.eupdate(means, covs, ob_m, ob_v, ob_f)
        if checks:
            for i, cov in enumerate(covs.view(-1, *covs.shape[-2:])):
                util.check_covariance(cov, f"step {t} elem {i} update")
        # Save post-update prediction.
        pred1.append(means[..., :K*D])
        covs1.append(covs[..., :K*D, :K*D])
        # Predict next state. (Only if not the last state.)
        # if t != T - 1:
        #     means, covs = model.predict_train(dt, means, covs)
        #     if checks:
        #         for i, cov in enumerate(covs.view(-1, *covs.shape[-2:])):
        #             util.check_covariance(cov, f"step {t} elem {i} predict")
    return torch.stack(pred0, dim=-2), torch.stack(covs0, dim=-3), \
        torch.stack(pred1, dim=-2), torch.stack(covs1, dim=-3)


def test_multivariate_normal_pairs(samples, means, covs):
    N, d = samples.shape
    total_mahalanobis_sq = 0.0
    for i in range(N):
        diff = samples[i] - means[i]
        inv_cov_diff = np.linalg.solve(covs[i], diff)
        d_sq = diff @ inv_cov_diff
        total_mahalanobis_sq += d_sq
    df = N * d
    p_value = chi2.sf(total_mahalanobis_sq, df=df)
    return float(total_mahalanobis_sq), float(p_value), float(df)


def test_hypothesis(track, sim_track, post_cov):
    return test_multivariate_normal_pairs(
        track.view(-1, 51).detach().cpu().numpy(),
        sim_track.view(-1, 51).detach().cpu().numpy(),
        post_cov.view(-1, 51, 51).detach().cpu().numpy()
    )


data = dataset.KalmanDataset("./data/kalman/train")
data_val = dataset.KalmanDataset("./data/kalman/val")
raw_data = dataset.CmuPanopticDataset(path="./data/panoptic")
cams = raw_data.get_some_cams()

print("Number of training samples:", len(data))
print("Number of validation samples:", len(data_val))

fps, track = data[int(sys.argv[1])]
print(f"FPS: {fps.item():.2f}")
track = track[-1].expand_as(track).contiguous()
track[..., 0] -= track[..., 0].mean()
track[..., 2] -= track[..., 2].mean()
# plot = visualize.MinimalSkeletonPlayer(track, fps.item())
model = kalman.LearnedPhysics(tracker.build_constrained_physics())
_, pre_cov, sim_track, post_cov = simulate_kalman_filter(
    model, fps, track, cams)
for pre, post, sim, gt in zip(pre_cov, post_cov, sim_track, track):
    print("::", pre.diag().sqrt().mean().tolist(), end=" ")
    print("=>", post.diag().sqrt().mean().tolist(), end=" ")
    print("-", test_hypothesis(gt, sim, post))
    # exit()
print("p-value", test_hypothesis(track, sim_track, post_cov))
plot = visualize.MinimalSkeletonPlayer(
    sim_track.detach().cpu().view_as(track), fps.item(), gt=track)
plot.setup_cameras(cams)
plot.show()
