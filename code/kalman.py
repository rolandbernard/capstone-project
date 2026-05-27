
from functools import lru_cache

import torch
import torch.nn as nn


class KalmanFilter(nn.Module):
    """
    This class implements a linear Kalman filter. The dynamics must be specified
    in continuous form, with the discretization computed and cached on demand.
    Both the prediction and the update step and in theory work on batches of targets.
    """

    def __init__(self, dyn_mat: torch.Tensor, dyn_cov: torch.Tensor):
        super().__init__()
        self.dyn_mat = nn.Parameter(dyn_mat)
        self.dyn_cov = nn.Parameter(dyn_cov)
        self.get_dyn = lru_cache()(self._get_dyn)

    def _get_dyn(self, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create a new dynamics and covariance matrix for the given timestamp.
        """
        # Use a second order approximation for now.
        N, N = self.dyn_mat.shape
        dyn_mat = torch.eye(N, device=self.dyn_mat.device) + self.dyn_mat * dt \
            + self.dyn_mat @ self.dyn_mat * (dt * dt * 0.5)
        dyn_cov = self.dyn_cov * dt + \
            (self.dyn_mat @ self.dyn_cov
             + self.dyn_cov @ self.dyn_mat.mT) * (dt * dt * 0.5)
        return dyn_mat, dyn_cov

    def predict(self, dt: float, mean: torch.Tensor, cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the internal prediction logic from this Kalman filter for the given
        amount of time having passed and return the new means and covariances for
        the targets.
        """
        if self.training:
            # Don't cache the result if we are in training mode.
            dyn_mat, dyn_cov = self._get_dyn(dt)
        else:
            dyn_mat, dyn_cov = self.get_dyn(dt)
        return kalman_predict(mean, cov, dyn_mat, dyn_cov)

    def update(
        self, mean: torch.Tensor, cov: torch.Tensor, obs_mat: torch.Tensor, obs_mean: torch.Tensor, obs_cov: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply an update given some observation. The observation is given using mean,
        covariance, and the matrix to extract it from the state. This API allow
        using different kinds of observation and uncertainties at different time
        steps.
        """
        return kalman_update(mean, cov, obs_mat, obs_mean, obs_cov)


single_batch_block_diag = torch.vmap(torch.block_diag)


def batched_block_diag(mats: list[torch.Tensor]) -> torch.Tensor:
    *Bs, _, _ = mats[0].shape
    batched = single_batch_block_diag(*[
        mat.view(-1, *mat.shape[-2:]) for mat in mats
    ])
    _, N, N = batched.shape
    return batched.view(*Bs, N, N)


def kalman_merge_obs(obs_mat: list[torch.Tensor], obs_mean: list[torch.Tensor], obs_cov: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Merge multiple observations into single matrix assuming the observations are
    independent. This allows multiple different observations to be performed with
    a single update step.
    """
    comb_mat = torch.concat(obs_mat, dim=-2)
    comb_mean = torch.concat(obs_mean, dim=-1)
    comb_cov = batched_block_diag(obs_cov)
    return comb_mat, comb_mean, comb_cov


def kalman_predict(mean: torch.Tensor, cov: torch.Tensor, dyn_mat: torch.Tensor, dyn_cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the prediction logic from this Kalman filter for the dynamics matrix
    and covariances having passed and return the new means and covariances for
    the targets.
    """
    return (dyn_mat @ mean.unsqueeze(-1)).squeeze(-1), dyn_mat @ cov @ dyn_mat.mT + dyn_cov


def kalman_update(
    mean: torch.Tensor, cov: torch.Tensor, obs_mat: torch.Tensor, obs_mean: torch.Tensor, obs_cov: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply an update given some observation. The observation is given using mean,
    covariance, and the matrix to extract it from the state.
    """
    res = obs_mean - (obs_mat @ mean.unsqueeze(-1)).squeeze(-1)
    inov = obs_mat @ cov @ obs_mat.mT + obs_cov
    gain = cov @ torch.linalg.solve(inov.mT, obs_mat).mT
    *_, N, N = gain.shape
    return (
        mean + (gain @ res.unsqueeze(-1)).squeeze(-1),
        (torch.eye(N, device=gain.device) - gain @ obs_mat) @ cov
    )
