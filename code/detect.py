
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import PoseModel


class PoseDetector:
    """
    This is an wrapper class around the YOLO based pose estimation models. Given
    an image it produces as output a tensor of shape (P, K, 3) where P is the
    number of detected persons an K = 17 is the number of keypoints.
    """

    def __init__(self, model_name: str = "yolo26n-pose", threshold: float = 0.25, path: str = "./nets"):
        model: PoseModel = YOLO(
            f"{path}/{model_name}.pt").model  # type: ignore
        model.eval()
        model.compile()
        self.model = model
        self.threshold = threshold
        self.num_keypoint = 17

    def detect(self, images: torch.Tensor | list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Run the detection algorithm and return discovered keypoints. We expect
        a batch of images and return a batch of results.
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
                valid = img_res[img_res[:, 4] > self.threshold, 6:]
                results.append(valid.view(-1, self.num_keypoint, 3))
            return results

    def to(self, *args, **kargs):
        """
        Apply the PyTorch `.to` method to the contained model.
        """
        self.model = self.model.to(*args, **kargs)
        return self
