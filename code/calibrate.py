
import configparser
from dataclasses import dataclass

import torch


@dataclass
class Camera:
    """
    A simple class to encapsulate all of the intrinsic and extrinsic camera
    calibration parameters necessary for mapping a 3d point into the camera
    perspective. Uses PyTorch tensors.
    """
    # Extrinsics
    rotation: torch.Tensor = torch.eye(3)
    translation: torch.Tensor = torch.zeros(3)
    # Intrinsics
    intrinsic: torch.Tensor = torch.eye(3)
    distortion: torch.Tensor = torch.zeros(5)

    def load_ini(self, path: str):
        """
        Load the calibration parameters from a file in a .ini format. This will
        replace the values in this instance with those in the file. If some
        sections are missing in the file, those parameters will simply be left
        as is.
        """
        config = configparser.ConfigParser()
        with open(path, "r") as file:
            config.read_file(file)
        if config.has_section("Extrinsics"):
            self.rotation = torch.tensor([
                [float(config["Extrinsics"][f"R{row + 1}{col + 1}"])
                 for col in range(3)]
                for row in range(3)
            ])
            self.translation = torch.tensor([
                float(config["Extrinsics"][f"T{row + 1}"]) for row in range(3)
            ])
        if config.has_section("Intrinsics"):
            section = config["Intrinsics"]
            f = float(section["f"])
            self.intrinsic = torch.tensor([
                [f * float(section["mu"]), 0.0, float(section["u0"])],
                [0.0, f * float(section["mv"]), float(section["v0"])],
                [0.0, 0.0, 1.0],
            ])
        if config.has_section("Distortion=pinhole"):
            section = config["Distortion=pinhole"]
            self.distortion = torch.tensor([
                float(section["k1"]), float(section["k2"]),
                float(section["p1"]), float(section["p2"]),
                float(section["k3"]),
            ])

    def save_ini(self, path: str, save_extrinsics=True):
        """
        Save the calibration parameters to a file in a .ini format. Optionally it
        is possible to not save the extrinsic parameters. This is useful in cases
        where the stored extrinsics are not meaningful.
        """
        config = configparser.ConfigParser()
        if save_extrinsics:
            config["Extrinsics"] = {}
            for row in range(3):
                for col in range(3):
                    config["Extrinsics"][f"R{row + 1}{col + 1}"] = str(
                        self.rotation[row, col].item())
            for row in range(3):
                config["Extrinsics"][f"T{row + 1}"] = str(
                    self.translation[row].item())
        f = self.intrinsic[0, 0].item()
        config["Intrinsics"] = {
            "f": str(f),
            "mu": str(1.0),
            "mv": str(self.intrinsic[1, 1].item() / f),
            "u0": str(self.intrinsic[0, 2].item()),
            "v0": str(self.intrinsic[1, 2].item()),
        }
        config["Distortion=pinhole"] = {
            n: str(self.distortion[i].item())
            for i, n in enumerate(["k1", "k2", "p1", "p2", "k3"])
        }
        with open(path, "w") as file:
          config.write(file)

    def project(self, points: torch.Tensor, eps=1e-7) -> torch.Tensor:
        """
        Project a set 3d points to 2d locations on the cameras image plane. The
        last dimensions of the input should be the points, and the others can be
        an arbitrarily number of batch dimensions. This is fully differentiable.
        """
        # Project into camera space
        points_cam = (points @ self.rotation.T) + self.translation
        xy, z = points_cam[..., 0:2], points_cam[..., 2:3]
        # Apply Tsai distortion
        z = torch.clamp(z, min=eps)
        xy_norm = xy / z
        k1, k2, _, _, k3 = self.distortion
        p12 = self.distortion[2:4]
        xy_norm_sqr = xy_norm**2
        r2 = torch.sum(xy_norm_sqr, dim=-1, keepdim=True)
        r4, r6 = r2**2, r2**3
        scale = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        xy_prod = torch.prod(xy_norm, dim=-1, keepdim=True)
        xy_off = 2.0 * p12 * xy_prod + \
            torch.flip(p12, [0]) * (r2 + 2.0 * xy_norm_sqr)
        xy_dist = xy_norm * scale + xy_off
        # Apply camera intrinsics
        uv = (xy_dist @ self.intrinsic[0:2, 0:2].T) + self.intrinsic[0:2, 2]
        return uv
    
    def project_pinhole(self, points: torch.Tensor, eps=1e-7) -> torch.Tensor:
        """
        Project a set 3d points to 2d locations on the cameras image plane. This
        compute ideal positions and does not take into acount camera intrinsics
        or distortion. This is fully differentiable.
        """
        points_cam = (points @ self.rotation.T) + self.translation
        xy, z = points_cam[..., 0:2], points_cam[..., 2:3]
        z = torch.clamp(z, min=eps)
        return xy / z


def triangulate(cams: list[Camera], points: list[torch.Tensor]) -> torch.Tensor:
    """
    Triangulate multiple points using multiple camera positions. All input tensors
    must have the same shape, with the last dimension having size 2 and an arbitrary
    number of batch dimensions in front. The output will have the same batch
    dimensions but a final dimension of size 3.
    """
    pass
