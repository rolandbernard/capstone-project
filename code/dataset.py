
import os

import gdown

import capture
from camera import Camera


class SalsaDataset:
    """
    Class for handling the Salsa dataset (https://tev.fbk.eu/resources/salsa).
    """

    def __init__(self, path: str = "./data/salsa"):
        """
        Create an instance of the class with the data stored in the given directory.
        """
        self.path = path

    def download(self):
        """
        Download the dataset, including video files and camera calibration
        parameters from Google Drive. Download is skipped if already present.
        """
        camera_ids = ["1DYHJoTZtKDzv7HIIZVGK3PChQi-XaQx1", "1JZT_SVSb1o1iOy3I6bzkcwId4WnH14UZ",
                      "164H1WzfinUbPiCuPNecK4_d6gyc9qStk", "1cUU9n1webV3OGJcvFD_xQN1tPWzlxYJr"]
        for i, id in enumerate(camera_ids):
            gdown.cached_download(  # type: ignore
                id=id, path=f"{self.path}/cam{i}.ini")
        poster_ids = ["1F-Q-t2UlGrK6GEl5Df0T72Sb_CxX60fg", "1_Vx1HO4UUBQMsBR8sk4LeiTUTZs4OB-4",
                      "1ZzGQH3CXFg6AbwkcMWkJYGTJ0qlldwRM", "1Kaviox0ZQHq0UYcIjkGIjp7z1fX7kNbh"]
        os.makedirs(f"{self.path}/PosterSession", exist_ok=True)
        for i, id in enumerate(poster_ids):
            gdown.cached_download(  # type: ignore
                id=id, path=f"{self.path}/PosterSession/cam{i}.avi")
        party_ids = ["18h6z8DzcFxbJVZeWuUK571bO710CwR9R", "1_0AQWAJiKV3E--Dki1mpLZPAdrU2JByg",
                     "1y02UrxsyLom_OkBwiQAJk7iTqU1tsZo2", "1panv-zguLPDquar5kmGo_akozMoNYpBe"]
        os.makedirs(f"{self.path}/CocktailParty", exist_ok=True)
        for i, id in enumerate(party_ids):
            gdown.cached_download(  # type: ignore
                id=id, path=f"{self.path}/CocktailParty/cam{i}.avi")

    def get_source(self, name: str = "PosterSession") -> capture.VideoSource:
        """
        Load one of the two video sequences from the dataset into a video source
        for further processing. The name can be wither "PosterSession" (default)
        or "CocktailParty".
        """
        streams = [f"{self.path}/{name}/cam{i}.avi" for i in range(4)]
        cameras = []
        for i in range(4):
            camera = Camera()
            camera.load_ini(f"{self.path}/cam{i}.ini")
            cameras.append(camera)
        return capture.OfflineVideoSource(streams, cameras)
