"""坐标系管理：world / base / camera / object / ee 之间的变换。

在本仿真中 Panda 基座 link0 固连于世界原点，因此 base 系与 world 系重合；
但仍以通用方式实现，便于迁移到基座有偏置的真实平台。
"""
from __future__ import annotations

import numpy as np

from .transforms import invert_transform, transform_from_quat_pos, transform_point


class FrameManager:
    """从 MuJoCo 状态读取各坐标系位姿，并提供坐标变换。"""

    def __init__(self, env, base_body: str = "link0", camera_name: str = "scene_cam"):
        self.env = env
        self.base_body = base_body
        self.camera_name = camera_name

    # -------- 基础位姿（世界系下） --------
    def T_world_base(self) -> np.ndarray:
        return self.env.body_pose(self.base_body)

    def T_world_cam(self) -> np.ndarray:
        """相机相对世界系的位姿（外参）。"""
        return self.env.camera_pose(self.camera_name)

    def T_base_cam(self) -> np.ndarray:
        """相机相对基座系（手眼外参，眼在手外配置）。"""
        return invert_transform(self.T_world_base()) @ self.T_world_cam()

    # -------- 常用变换 --------
    def cam_to_world(self, point_cam: np.ndarray) -> np.ndarray:
        return transform_point(self.T_world_cam(), point_cam)

    def cam_to_base(self, point_cam: np.ndarray) -> np.ndarray:
        return transform_point(self.T_base_cam(), point_cam)

    def world_to_base(self, point_world: np.ndarray) -> np.ndarray:
        return transform_point(invert_transform(self.T_world_base()), point_world)
