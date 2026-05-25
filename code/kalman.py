
import torch


class KalmanFiler:
    """
    This class implements a linear Kalman filter. The dynamics must be specified
    in continuous form, with the discretization computed and cached on demand.
    Both the prediction and the update step and in theory work on batches of targets.
    """

    def predict(self, dt: float, mean: torch.Tensor, cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the internal production logic from this Kalman filter for the given
        amount of time having passed and return the new means and covariances for
        the targets.
        """
        pass

    def update(
        self, mean: torch.Tensor, cov: torch.Tensor, obs_mat: torch.Tensor, obs_mean: torch.Tensor, obs_cov: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply an update given some observation. The observation is given using mean,
        covariance, and the matrix to extract it from the state. This API allow
        using different kinds of observation and uncertainties at different time
        steps.
        """
        pass
