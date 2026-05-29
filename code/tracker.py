
import torch

from camera import Camera
from detect import PoseDetector
from kalman import LinearPhysics


class Track:
    """
    This is a simple implementation of the track in which the state is made up
    of a vector in which the first elements form the coordinates of keypoints and
    a matrix corresponding to the complete covariance matrix. This  type of track
    does not include history.
    """

    def __init__(self, id: int, init_mean: torch.Tensor, init_cov: torch.Tensor, num_keypoint=17, num_dim=3):
        self.id = id
        self.num_detection = 0
        self.num_keypoint = num_keypoint
        self.num_dim = num_dim
        self.update(init_mean, init_cov)

    def moved(self, mean: torch.Tensor, cov: torch.Tensor):
        """
        Update the track to the new state (from a prediction).
        """
        self.mean = mean
        self.cov = cov

    def update(self, mean: torch.Tensor, cov: torch.Tensor):
        """
        Update the track to the new state (from a detection).
        """
        self.moved(mean, cov)
        self.last_detection = 0
        self.num_detection += 1

    def no_update(self):
        """
        Record that there was no update for this track for one update cycle.
        """
        self.last_detection += 1

    def get_keypoints(self) -> torch.Tensor:
        """
        Get the keypoints for this track. The keypoints should be derived from
        the internal state of the track in some implementation defined way.
        """
        return self.mean.view(-1, self.num_dim)[:self.num_keypoint]

    def get_full_covariances(self) -> torch.Tensor:
        """
        Get a full covariance matrix for this track, including covariances between
        different keypoints. This is used internally, but for visualization we
        use `get_covariances`.
        """
        tot = self.num_keypoint*self.num_dim
        return self.cov[:tot, :tot].view(-1, self.num_dim, self.num_keypoint, self.num_dim)

    def get_covariances(self) -> torch.Tensor:
        """
        Get a covariance matrix for each of the keypoints. The covariances are
        marginalized for each keypoint even if there are inter-keypoint variances.
        May return a small diagonal matrix if not implemented.
        """
        blocks = torch.diagonal(self.get_full_covariances(), dim1=0, dim2=2)
        return blocks.permute(2, 0, 1)


class Tracker:
    """
    A tracker that only tracks objects in individual 2d images. Detections are
    matched to tracks using the hungarian algorithm. Tracks become active if they
    are observed a sufficient number of times, and removed once they are not
    observed for some time.
    """

    def __init__(self, detector: PoseDetector, physics: LinearPhysics, min_age: int = 3, max_inv: int = 30):
        self.tracks: list[Track] = []
        self.last_id = 0
        self.detector = detector
        self.physics = physics
        self.min_age = min_age
        self.max_inv = max_inv

    def get_prediction(self, dt: None | float = None) -> list[Track]:
        """
        Get the internal state prediction for the given time in the future. By
        default, if no time step is given, the current prediction is returned.
        """
        if ts == self.last_ts:
            # We don't really predict, we assume everything stays the same.
            return [track for track in self.tracks if track.num_detection > self.min_age]
        else:
            pass  # TODO

    def predict(self, dt: float):
        """
        Process a forward step in time to the given amount of seconds and update
        the internal state of all tracks accordingly, but without using any new
        external information.
        """
        means = torch.stack([track.mean for track in self.tracks])
        covs = torch.stack([track.cov for track in self.tracks])
        means, covs = self.physics.predict(dt, means, covs)
        for track, mean, cov in zip(self.tracks, means, covs):
            track.moved(mean, cov)

    def associate_pred_to_detection(self, cam: Camera, detections: torch.Tensor) -> list[torch.Tensor]:
        """
        Associate the given detections to one of the existing tracks. At most
        one detection in associated to each track. Returns two tensors of shape
        where the first has index of the tracks.
        """
        pass

    def associate_detections(self, cams: list[Camera], detections: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Associate the given detections to one of the existing tracks. At most
        one detection in associated to each track. Returns tensors with index
        into the given detections array.
        """
        pass

    def update(self, cams: list[Camera], imgs: list[torch.Tensor]):
        """
        Update the internal track states based on new incoming images, but don't
        perform any internal time step updates.
        """
        detections = self.detector.detect(cams, imgs)

        # prediction = torch.stack([track.state.view(-1, 3)
        #                          for track in self.tracks])
        # detections = self.detector.detect(imgs)[0]
        # cost_mat = prediction[:, None, :, 0:2] - detections[None, :, :, 0:2]
        # cost_mat = cost_mat*cost_mat
        # visible = prediction[:, None, :, 2:3] * detections[None, :, :, 2:3]
        # cost_mat = (cost_mat * visible).sum(dim=(2, 3)) / visible.sum(dim=-1)
