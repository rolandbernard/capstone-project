
import torch

from camera import Camera
from detect import PoseDetector

class Track:
    """
    An abstract class to represent a track of a single object in the tracking system.
    """

    def __init__(self, init_state):
        self.last_detection = 0
        self.num_detection = 0
        self.state = init_state

    def get_keypoints(self) -> torch.Tensor:
        """
        Get the keypoints for this track. The keypoints should be derived from
        the internal state of the track in some implementation defined way.
        """
        raise NotImplementedError

    def get_covariances(self) -> torch.Tensor:
        """
        Get a covariance matrix for each of the keypoints. The covariances are
        marginalized for each keypoint even if there are inter-keypoint variances.
        May return a small diagonal matrix if not implemented.
        """
        K, D = self.get_keypoints().shape
        return torch.eye(D).expand(K, D, D)


class Tracker:
    """
    Abstract class for implementing the main logic of the tracking system. The
    tracker is given as input timestamps, images, and camera locations. From
    these it tries to reconstruct the individual objects positions in each frame.
    """

    def __init__(self):
        self.tracks: list[Track] = []
        self.last_ts = 0

    def get_prediction(self, ts: None | float = None) -> list[Track]:
        """
        Get the internal state prediction for the given time in the future. By
        default, if no time step is given, the current prediction is returned.
        """
        raise NotImplementedError

    def predict_and_update(self, ts: float, cams: list[Camera], imgs: list[torch.Tensor]):
        """
        This is a combination of the internal prediction and update using new
        images from the given cameras. The internal state is advanced to `ts`
        seconds and then updated using the new information in the images.
        """
        raise NotImplementedError

    def predict(self, ts: float):
        """
        Process a forward step in time to the given amount of seconds and update
        the internal state of all tracks accordingly, but without using any new
        external information.
        """
        return self.predict_and_update(ts, [], [])

    def update(self, cams: list[Camera], imgs: list[torch.Tensor]):
        """
        Update the internal track states based on new incoming images, but don't
        perform any internal time step updates.
        """
        return self.predict_and_update(self.last_ts, cams, imgs)


class SimpleTrack(Track):
    """
    This is a simple implementation of the track in which the state is a single
    vector in which the first elements form the coordinates of keypoints. This 
    type of track does not have covariances.
    """

    def __init__(self, init_state: torch.Tensor, num_keypoint: int = 17, num_dim: int = 3):
        super().__init__(init_state)
        self.num_keypoint = num_keypoint
        self.num_dim = num_dim

    def get_keypoints(self) -> torch.Tensor:
        return self.state[:self.num_keypoint * self.num_dim] \
            .view(self.num_keypoint, self.num_dim)


class Simple2dTracker(Tracker):
    """
    A simple tracker that only tracks objects in individual 2d images without
    any prediction. Detections are matched to tracks using the hungarian algorithm.
    Tracks become active if they are observed a sufficient number of times, and
    removed once they are not observed for some time.
    """

    def __init__(self, detector: PoseDetector, min_age: int = 3, max_inv: int = 30):
        super().__init__()
        self.min_age = min_age
        self.max_inv = max_inv

    def get_prediction(self, ts: None | float = None) -> list[Track]:
        # We don't really predict, we assume everything stays the same.
        return [track for track in self.tracks if track.num_detection > self.min_age]

    def predict_and_update(self, ts: float, cams: list[Camera], imgs: list[torch.Tensor]):
        # No prediction.
