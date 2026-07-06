"""RGB-D 相机：离屏渲染、内参解析、像素反投影。"""
from __future__ import annotations

import mujoco
import numpy as np

from ..geometry.transforms import transform_point


class RGBDCamera:
    """基于 MuJoCo 离屏渲染的 RGB-D 相机。

    同时输出 RGB 图与深度图，并由 fovy 与分辨率解析内参矩阵 K，
    支持像素 (u,v,depth) -> 相机系 -> 世界系的反投影。
    """

    def __init__(self, env, camera_name: str, width: int = 640, height: int = 480):
        self.env = env
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self._renderer = mujoco.Renderer(env.model, height=height, width=width)
        self._depth_renderer = mujoco.Renderer(env.model, height=height, width=width)
        self._depth_renderer.enable_depth_rendering()
        self.K = self._compute_intrinsics()

    def _compute_intrinsics(self) -> np.ndarray:
        """由竖直视场角 fovy 与分辨率计算针孔相机内参矩阵。"""
        fovy = np.deg2rad(self.env.camera_fovy(self.camera_name))
        f = 0.5 * self.height / np.tan(fovy / 2.0)  # 竖直方向焦距（像素）
        cx = (self.width - 1) / 2.0
        cy = (self.height - 1) / 2.0
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=float)

    def render(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 (rgb[H,W,3] uint8, depth[H,W] float32, 单位 m)。"""
        self._renderer.update_scene(self.env.data, camera=self.camera_name)
        rgb = self._renderer.render()
        self._depth_renderer.update_scene(self.env.data, camera=self.camera_name)
        depth = self._depth_renderer.render()
        return rgb, depth

    # ----------------------------------------------------------------- 反投影
    def pixel_to_camera(self, u: float, v: float, depth: float) -> np.ndarray:
        """像素 (u,v) + 深度 -> 相机坐标系 3D 点。

        采用视觉常用相机系：x 右、y 下、z 朝前（指向场景）。
        """
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        z = float(depth)
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z])

    def camera_to_world(self, point_cam_cv: np.ndarray) -> np.ndarray:
        """相机系(视觉约定 x右y下z前) 点 -> 世界系。

        MuJoCo 相机系为 x右、y上、z朝外(看向 -z)，与视觉约定相差绕 x 轴 180°，
        即 (x, y, z)_cv -> (x, -y, -z)_mj。
        """
        p_mj = np.array([point_cam_cv[0], -point_cam_cv[1], -point_cam_cv[2]])
        T_world_cam = self.env.camera_pose(self.camera_name)
        return transform_point(T_world_cam, p_mj)

    def pixel_to_world(self, u: float, v: float, depth: float) -> np.ndarray:
        return self.camera_to_world(self.pixel_to_camera(u, v, depth))

    def pixel_ray_world(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        """返回像素 (u,v) 对应的世界系相机光线 (原点, 单位方向)。"""
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        d_cv = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        d_mj = np.array([d_cv[0], -d_cv[1], -d_cv[2]])  # cv -> mujoco 相机系
        T = self.env.camera_pose(self.camera_name)
        origin = T[:3, 3]
        direction = T[:3, :3] @ d_mj
        direction = direction / np.linalg.norm(direction)
        return origin, direction

    def pixel_to_plane(self, u: float, v: float, z_plane: float) -> np.ndarray:
        """像素光线与已知水平支撑平面 z=z_plane 的交点（世界系）。

        利用目标位于已知高度平面这一先验，可消除深度噪声对水平定位的影响。
        """
        origin, direction = self.pixel_ray_world(u, v)
        if abs(direction[2]) < 1e-6:
            return self.pixel_to_world(u, v, 1.0)
        t = (z_plane - origin[2]) / direction[2]
        return origin + t * direction
