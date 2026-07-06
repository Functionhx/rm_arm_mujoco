"""夹爪开合控制与抓取判据。

夹爪视觉上正常闭合；当指定物体位于指间（TCP 附近）时，同步激活焊接约束以保证
牢固夹持（等效真实夹爪的稳定抓握），松开时解除约束。
"""
from __future__ import annotations

import numpy as np


class GripperController:
    def __init__(self, env):
        self.env = env
        g = env.robot_cfg["gripper"]
        self.width_open = g["width_open"]

    def open(self, steps: int = 40, recorder=None) -> None:
        self.env.detach_object()
        self.env.set_gripper(open_=True)
        for _ in range(steps):
            self.env.step_ctrl_cycle()
            if recorder is not None:
                recorder.tick()

    def close(self, steps: int = 60, recorder=None,
              attach_body: str | None = None, attach_tol: float = 0.06) -> bool:
        """闭合夹爪。若 attach_body 位于 TCP 附近，则在闭合后激活焊接约束。

        返回是否成功夹持（物体被约束或指间仍有物体）。
        """
        self.env.set_gripper(open_=False)
        for _ in range(steps):
            self.env.step_ctrl_cycle()
            if recorder is not None:
                recorder.tick()
        grasped = self.is_grasping()
        if attach_body is not None:
            tcp = self.env.ee_pose()[:3, 3]
            obj = self.env.body_position(attach_body)  # 物体 body 原点（底部）
            xy = float(np.linalg.norm(tcp[:2] - obj[:2]))
            dz = float(tcp[2] - obj[2])
            # TCP 水平对准物体、且落在物体竖直高度范围内，则视为夹住并激活焊接
            if xy < attach_tol and -0.02 < dz < 0.16:
                self.env.attach_object(obj_body=attach_body)
                grasped = True
        return grasped

    def is_grasping(self, min_width: float = 0.004) -> bool:
        """夹爪未完全闭合（指间仍有物体）或已激活焊接约束，视为夹持成功。"""
        return self.env.is_object_attached() or self.env.gripper_width() > min_width
