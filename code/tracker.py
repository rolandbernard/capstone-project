
import torch

from code.camera import Camera


class Track:
    def __init__(self, init_state) -> None:
       self.last_detection = 0
       self.state = init_state

    def get_keypoints(self) -> torch.Tensor:
        pass

    def get_covariances(self) -> torch.Tensor:
        pass


class Tracker:
    def __init__(self) -> None:
       self.tracks: list[Track] = []
       self.entering = []

    def get_prediction(self, dt: float = 0.0) -> list[Track]:
        """
        Get the internal state prediction for the given time in the future. By
        default, if no time step is given, the current prediction is returned.
        """
        if dt == 0.0:
            return self.tracks
        else:
            pass

    def predict_and_update(self, dt: float, cams: list[Camera], imgs: list[torch.Tensor]):
        """
        This is a combination of the internal prediction and update using new
        images from the given cameras. The internal state is advanced by `dt`
        seconds and then updated using the new information in the images.
        """
        pass

    def predict(self, dt: float):
        """
        Process a forward step in time by the given amount of seconds and update
        the internal state of all tracks accordingly, but without using any new
        external information.
        """
        return self.predict_and_update(dt, [], [])

    def update(self, cams: list[Camera], imgs: list[torch.Tensor]):
        """
        Update the internal track states based on new incoming images, but don't
        perform any internal time step updates.
        """
        return self.predict_and_update(0.0, cams, imgs)
