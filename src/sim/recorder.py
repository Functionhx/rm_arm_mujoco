"""录制：视频帧采集、截图、轨迹记录与保存。"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np


class Recorder:
    def __init__(self, env, camera: str = "video_cam", width: int = 800, height: int = 600,
                 fps: int = 30, stride: int = 2):
        self.env = env
        self.camera = camera
        self.width = width
        self.height = height
        self.fps = fps
        self.stride = max(1, stride)
        self._renderer = mujoco.Renderer(env.model, height=height, width=width)
        self.frames: list[np.ndarray] = []
        self.ee_traj: list[np.ndarray] = []
        self.q_traj: list[np.ndarray] = []
        self.t_traj: list[float] = []
        self._count = 0
        # 关键阶段截图（进入这些状态时各截一张，用于报告分阶段配图）
        self.stage_snaps: dict[str, np.ndarray] = {}
        self._stage_on_entry = ["DESCEND", "GRASP", "LIFT", "INSERT", "RELEASE", "DONE"]

    def on_state(self, name: str) -> None:
        if name in self._stage_on_entry and name not in self.stage_snaps:
            self.stage_snaps[name] = self.snapshot()

    def tick(self) -> None:
        """在控制循环中调用：按 stride 采集一帧并记录当前末端/关节状态。"""
        self.ee_traj.append(self.env.ee_pose()[:3, 3].copy())
        self.q_traj.append(self.env.get_arm_qpos())
        self.t_traj.append(self.env.time)
        if self._count % self.stride == 0:
            self._renderer.update_scene(self.env.data, camera=self.camera)
            self.frames.append(self._renderer.render())
        self._count += 1

    def snapshot(self, camera: str | None = None) -> np.ndarray:
        self._renderer.update_scene(self.env.data, camera=camera or self.camera)
        return self._renderer.render()

    def save_screenshot(self, path: str, camera: str | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(path, self.snapshot(camera))

    def save_video(self, path: str) -> None:
        if not self.frames:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(path, self.frames, fps=self.fps, macro_block_size=None)

    def save_trajectory(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            t=np.array(self.t_traj),
            ee=np.array(self.ee_traj),
            q=np.array(self.q_traj),
        )
