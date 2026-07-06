"""机械臂关节位置控制：跟踪关节轨迹 / 笛卡尔直线运动。

Panda 使用位置型执行器（affine bias 等效 PD），设 ctrl 为目标关节角即可驱动到位。
"""
from __future__ import annotations

import numpy as np

from ..geometry.transforms import make_transform
from ..planning.trajectory import cartesian_line, joint_trajectory


class ArmController:
    def __init__(self, env, ik):
        self.env = env
        self.ik = ik
        sm = env.robot_cfg["ik"]
        self.pos_tol = sm["tol_pos"]

    # ------------------------------------------------------------- 关节空间
    def move_to_q(self, q_goal: np.ndarray, num: int = 60, hold: int = 15,
                  recorder=None) -> None:
        """沿五次多项式关节轨迹平滑运动到 q_goal。"""
        q0 = self.env.get_arm_qpos()
        traj = joint_trajectory(q0, q_goal, num)
        for q in traj:
            self.env.set_arm_ctrl(q)
            self.env.step_ctrl_cycle()
            if recorder is not None:
                recorder.tick()
        # 稳定
        for _ in range(hold):
            self.env.step_ctrl_cycle()
            if recorder is not None:
                recorder.tick()

    # ------------------------------------------------------------- 笛卡尔直线
    def move_cartesian(self, target_pose: np.ndarray, num: int = 40, hold: int = 15,
                       recorder=None) -> tuple[bool, float]:
        """末端沿笛卡尔直线运动到 target_pose（保持姿态为目标姿态）。

        对每个路径点求 IK，逐点位置控制，实现末端近似直线、平稳的运动。
        """
        T_cur = self.env.ee_pose()
        p_start = T_cur[:3, 3]
        p_goal = target_pose[:3, 3]
        R_goal = target_pose[:3, :3]
        pts = cartesian_line(p_start, p_goal, num)
        q_seed = self.env.get_arm_qpos()
        ok_all = True
        for p in pts:
            Tw = make_transform(R_goal, p)
            q, ok, err = self.ik.solve(Tw, q_init=q_seed)
            q_seed = q
            ok_all = ok_all and ok
            self.env.set_arm_ctrl(q)
            self.env.step_ctrl_cycle()
            if recorder is not None:
                recorder.tick()
        for _ in range(hold):
            self.env.step_ctrl_cycle()
            if recorder is not None:
                recorder.tick()
        final_err = float(np.linalg.norm(self.env.ee_pose()[:3, 3] - p_goal))
        return ok_all, final_err

    def go_pose(self, target_pose: np.ndarray, num: int = 60, hold: int = 15,
                recorder=None) -> tuple[bool, float]:
        """关节空间运动到某位姿（用于大范围移动，不要求末端直线）。"""
        q, ok, err = self.ik.solve(target_pose, q_init=self.env.get_arm_qpos())
        self.move_to_q(q, num=num, hold=hold, recorder=recorder)
        final_err = float(np.linalg.norm(self.env.ee_pose()[:3, 3] - target_pose[:3, 3]))
        return ok, final_err
