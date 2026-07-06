"""位姿估计：融合颜色检测 + RGB-D 反投影，输出目标在世界/基座系的位姿。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .color_detector import ColorDetector, Detection


@dataclass
class ObjectPose:
    found: bool
    position_world: np.ndarray = None   # 世界系 xyz
    position_base: np.ndarray = None    # 基座系 xyz
    pixel: tuple = (0.0, 0.0)
    detection: Detection = None


class PoseEstimator:
    """统一输出 ObjectPose。

    x,y 由质心像素反投影得到；z 结合反投影结果与桌面/物体先验取稳定值。
    """

    def __init__(self, camera, frames, vision_cfg: dict, table_top: float = 0.40):
        self.camera = camera
        self.frames = frames
        self.detector = ColorDetector(vision_cfg)
        self.table_top = table_top

    def estimate(self, rgb: np.ndarray, depth: np.ndarray) -> ObjectPose:
        det = self.detector.detect(rgb)
        if not det.found:
            return ObjectPose(found=False, detection=det)
        u, v = int(round(det.u)), int(round(det.v))
        u = np.clip(u, 0, depth.shape[1] - 1)
        v = np.clip(v, 0, depth.shape[0] - 1)
        # 取质心邻域中位深度，抑制噪声
        d = self._robust_depth(depth, u, v, r=3)
        p_depth = self.camera.pixel_to_world(u, v, d)  # 纯深度反投影（含 z）
        # 水平定位：用相机光线与已知抓取高度平面求交，消除深度噪声引入的 xy 偏差
        z_plane = self.table_top + 0.07
        p_plane = self.camera.pixel_to_plane(u, v, z_plane)
        p_world = np.array([p_plane[0], p_plane[1], p_depth[2]])
        p_base = self.frames.world_to_base(p_world)
        return ObjectPose(
            found=True,
            position_world=p_world,
            position_base=p_base,
            pixel=(det.u, det.v),
            detection=det,
        )

    @staticmethod
    def _robust_depth(depth: np.ndarray, u: int, v: int, r: int = 3) -> float:
        v0, v1 = max(0, v - r), min(depth.shape[0], v + r + 1)
        u0, u1 = max(0, u - r), min(depth.shape[1], u + r + 1)
        patch = depth[v0:v1, u0:u1].reshape(-1)
        patch = patch[np.isfinite(patch)]
        return float(np.median(patch)) if patch.size else float(depth[v, u])
