"""正运动学：给定手臂关节角，计算末端执行器 (TCP) 位姿。

使用一份 scratch MjData，避免扰动主仿真状态。
"""
from __future__ import annotations

import mujoco
import numpy as np

from ..geometry.transforms import make_transform


class ForwardKinematics:
    def __init__(self, env):
        self.env = env
        self.model = env.model
        self._scratch = mujoco.MjData(self.model)
        self._site_id = env._ee_site_id
        self._qadr = env._arm_qpos_adr

    def compute(self, q_arm: np.ndarray) -> np.ndarray:
        """返回 TCP 在世界系的 4x4 位姿。"""
        # 以当前完整状态为基准（保留物体位置），仅改写手臂关节
        self._scratch.qpos[:] = self.env.data.qpos
        self._scratch.qpos[self._qadr] = q_arm
        mujoco.mj_kinematics(self.model, self._scratch)
        R = self._scratch.site_xmat[self._site_id].reshape(3, 3)
        p = self._scratch.site_xpos[self._site_id]
        return make_transform(R, p)
