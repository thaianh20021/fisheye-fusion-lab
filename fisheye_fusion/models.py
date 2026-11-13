from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
from .geometry import rigid_transform


CameraModel = Literal["pinhole", "opencv_pinhole", "opencv_fisheye", "mei"]


class CameraConfig(BaseModel):
    id: str
    model: CameraModel
    resolution: tuple[int, int]
    K: list[list[float]]
    D: list[float] = Field(default_factory=list)
    xi: float | None = None
    T_rig_from_camera: list[list[float]]

    @model_validator(mode="after")
    def check_calibration(self):
        k = np.asarray(self.K)
        if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
            raise ValueError("K must be finite 3x3 with positive focal lengths")
        if not np.allclose(k[2], [0, 0, 1]) or k[0, 1] != 0 or k[1, 0] != 0:
            raise ValueError("K requires zero skew and bottom row [0,0,1]")
        if not np.isfinite(self.D).all() or len(self.D) not in ((4,) if self.model in ("opencv_fisheye", "mei") else (0, 4, 5, 8, 12, 14)):
            raise ValueError("invalid distortion coefficients")
        if self.model == "mei":
            if self.xi is None or not np.isfinite(self.xi) or self.xi < 0:
                raise ValueError("Mei camera requires finite nonnegative xi")
        elif self.xi is not None:
            raise ValueError("xi is only supported for Mei cameras")
        rigid_transform(self.T_rig_from_camera)
        return self

    @field_validator("resolution")
    @classmethod
    def valid_resolution(cls, value: tuple[int, int]) -> tuple[int, int]:
        if min(value) <= 0 or max(value) > 8192:
            raise ValueError("resolution must be between 1 and 8192")
        return value


class FrameConfig(BaseModel):
    timestamp_ns: int = Field(ge=0)
    images: dict[str, str]
    image_timestamps_ns: dict[str, int] = Field(default_factory=dict)
    T_world_from_rig: list[list[float]]
    lidar: str | None = None

    @model_validator(mode="after")
    def check_image_timestamps(self):
        if self.image_timestamps_ns:
            if set(self.image_timestamps_ns) != set(self.images):
                raise ValueError("image timestamps must cover exactly the frame images")
            if any(value < 0 for value in self.image_timestamps_ns.values()):
                raise ValueError("image timestamps must be nonnegative")
        return self

    @field_validator("T_world_from_rig")
    @classmethod
    def valid_pose(cls, value):
        rigid_transform(value)
        return value


class FusionConfig(BaseModel):
    camera_ids: list[str]
    output: Literal["equirectangular"] = "equirectangular"
    resolution: tuple[int, int] = (1024, 512)

    @field_validator("resolution")
    @classmethod
    def bounded_output(cls, value):
        if min(value) < 16 or max(value) > 2048:
            raise ValueError("preview resolution must be 16..2048")
        return value


class ProjectManifest(BaseModel):
    schema_version: Literal[1] = 1
    project: str
    root: str = "."
    cameras: list[CameraConfig]
    frames: list[FrameConfig]
    fusion: FusionConfig

    @model_validator(mode="after")
    def check_references(self):
        ids = [camera.id for camera in self.cameras]
        selected = self.fusion.camera_ids
        if len(ids) != len(set(ids)):
            raise ValueError("camera IDs must be unique")
        if len(selected) not in (1, 2, 4) or len(set(selected)) != len(selected):
            raise ValueError("select 1, 2, or 4 unique cameras")
        if not set(selected).issubset(ids):
            raise ValueError("unknown selected camera")
        if not self.frames:
            raise ValueError("frames must not be empty")
        times = [f.timestamp_ns for f in self.frames]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("timestamps must be strictly increasing")
        for frame in self.frames:
            if not set(frame.images).issubset(ids) or not set(selected).issubset(frame.images):
                raise ValueError("missing or unknown camera image reference")
        return self
