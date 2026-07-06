"""逆运动学：阻尼最小二乘 (DLS) 雅可比迭代法。

Δq = J^T (J J^T + λ^2 I)^{-1} e
在 scratch MjData 上迭代，不扰动主仿真；支持关节限位裁剪。
"""
from __future__ import annotations

import mujoco
import numpy as np

from ..geometry.transforms import make_transform, pose_error
from .jacobian import site_jacobian


class IKSolver:
    def __init__(self, env):
        self.env = env
        self.model = env.model
        self._scratch = mujoco.MjData(self.model)
        self._site_id = env._ee_site_id
        self._qadr = env._arm_qpos_adr
        self._dofadr = env._arm_dof_adr

        cfg = env.robot_cfg["ik"]
        self.max_iters = cfg["max_iters"]
        self.tol_pos = cfg["tol_pos"]
        self.tol_rot = cfg["tol_rot"]
        self.damping = cfg["damping"]
        self.step_scale = cfg["step_scale"]

        # 关节限位
        jids = [env._jid(j) for j in env.arm_joints]
        self.q_lower = self.model.jnt_range[jids, 0].copy()
        self.q_upper = self.model.jnt_range[jids, 1].copy()

    def _fk(self, q_arm: np.ndarray) -> np.ndarray:
        self._scratch.qpos[self._qadr] = q_arm
        mujoco.mj_kinematics(self.model, self._scratch)
        mujoco.mj_comPos(self.model, self._scratch)  # 雅可比所需
        R = self._scratch.site_xmat[self._site_id].reshape(3, 3)
        p = self._scratch.site_xpos[self._site_id]
        return make_transform(R, p)

    def solve(
        self,
        target_pose: np.ndarray,
        q_init: np.ndarray | None = None,
        pos_only: bool = False,
    ) -> tuple[np.ndarray, bool, float]:
        """求解到达 target_pose(4x4) 的关节角。

        返回 (q_solution[7], success, final_pos_error)。
        """
        # 以当前完整状态为基准，保留物体等其它自由度
        self._scratch.qpos[:] = self.env.data.qpos
        q = (q_init if q_init is not None else self.env.get_arm_qpos()).astype(float).copy()

        lam2 = self.damping ** 2
        last_pos_err = 1e9
        for _ in range(self.max_iters):
            T_cur = self._fk(q)
            err6 = pose_error(T_cur, target_pose)
            if pos_only:
                err6[3:] = 0.0
            pos_err = np.linalg.norm(err6[:3])
            rot_err = np.linalg.norm(err6[3:])
            last_pos_err = pos_err
            if pos_err < self.tol_pos and (pos_only or rot_err < self.tol_rot):
                return self._clamp(q), True, pos_err

            J = site_jacobian(self.model, self._scratch, self._site_id, self._dofadr)
            if pos_only:
                J = J[:3, :]
                e = err6[:3]
            else:
                e = err6
            # DLS
            JJt = J @ J.T
            dq = J.T @ np.linalg.solve(JJt + lam2 * np.eye(JJt.shape[0]), e)
            q = q + self.step_scale * dq
            q = self._clamp(q)

        return self._clamp(q), (last_pos_err < self.tol_pos * 2.0), last_pos_err

    def _clamp(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.q_lower, self.q_upper)
