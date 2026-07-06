"""MuJoCo 仿真环境封装：加载模型、step、reset、读取状态、位姿查询。"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import yaml

from ..geometry.transforms import make_transform, transform_from_quat_pos
from ..utils.logger import get_logger

log = get_logger("sim")


def _project_root() -> Path:
    # src/sim/mujoco_env.py -> 项目根
    return Path(__file__).resolve().parents[2]


def load_yaml(rel_path: str) -> dict:
    with open(_project_root() / rel_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class MujocoEnv:
    """封装 MjModel / MjData 及常用操作。"""

    def __init__(self, sim_cfg: str = "configs/sim.yaml", robot_cfg: str = "configs/robot.yaml"):
        self.sim_cfg = load_yaml(sim_cfg)
        self.robot_cfg = load_yaml(robot_cfg)

        scene_path = _project_root() / self.sim_cfg["scene"]
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)

        self.control_hz = self.sim_cfg["sim"]["control_hz"]
        self.dt = self.model.opt.timestep
        self.steps_per_ctrl = max(1, int(round((1.0 / self.control_hz) / self.dt)))

        r = self.robot_cfg["robot"]
        self.arm_joints = r["arm_joints"]
        self.ee_site = r["ee_site"]
        self.base_body = r["base_body"]
        self.home_qpos = np.array(r["home_qpos"], dtype=float)

        g = self.robot_cfg["gripper"]
        self.gripper_actuator = g["actuator"]
        self.grip_open = g["ctrl_open"]
        self.grip_close = g["ctrl_close"]

        # 关节 qpos / dof / actuator 索引
        self._arm_qpos_adr = np.array(
            [self.model.jnt_qposadr[self._jid(j)] for j in self.arm_joints]
        )
        self._arm_dof_adr = np.array(
            [self.model.jnt_dofadr[self._jid(j)] for j in self.arm_joints]
        )
        self._arm_act_ids = np.array(
            [self._aid(a) for a in r["arm_actuators"]]
        )
        self._grip_act_id = self._aid(self.gripper_actuator)
        self._ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site)

    # ----------------------------------------------------------------- id 帮助
    def _jid(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def _aid(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def _bid(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def _sid(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)

    # ----------------------------------------------------------------- 复位 / 步进
    def reset(self, settle: bool = True) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.set_arm_qpos(self.home_qpos)
        self.set_arm_ctrl(self.home_qpos)
        self.set_gripper(open_=True)
        mujoco.mj_forward(self.model, self.data)
        if settle:
            self.step(self.sim_cfg["sim"]["settle_steps"])

    def step(self, n: int = 1) -> None:
        for _ in range(int(n)):
            # 仅对手臂关节做重力/科氏力补偿（等效真实机械臂的重力补偿），
            # 消除位置控制在负载下的稳态下垂；自由物体不补偿，仍受重力与接触作用。
            self.data.qfrc_applied[self._arm_dof_adr] = self.data.qfrc_bias[self._arm_dof_adr]
            mujoco.mj_step(self.model, self.data)

    def step_ctrl_cycle(self) -> None:
        """按控制频率步进一个控制周期。"""
        self.step(self.steps_per_ctrl)

    # ----------------------------------------------------------------- 关节状态
    def get_arm_qpos(self) -> np.ndarray:
        return self.data.qpos[self._arm_qpos_adr].copy()

    def get_arm_qvel(self) -> np.ndarray:
        return self.data.qvel[self._arm_dof_adr].copy()

    def set_arm_qpos(self, q: np.ndarray) -> None:
        self.data.qpos[self._arm_qpos_adr] = q

    def set_arm_ctrl(self, q: np.ndarray) -> None:
        self.data.ctrl[self._arm_act_ids] = q

    def set_gripper(self, open_: bool) -> None:
        self.data.ctrl[self._grip_act_id] = self.grip_open if open_ else self.grip_close

    # ----------------------------------------------------------------- 抓取焊接约束
    def _weld_id(self) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld")

    def attach_object(self, hand_body: str = "hand", obj_body: str = "energy_unit") -> None:
        """在当前相对位姿下激活 hand<->obj 焊接约束，等效牢固夹持。"""
        eid = self._weld_id()
        if eid < 0:
            return
        bh = self._bid(hand_body)
        bo = self._bid(obj_body)
        # 计算 obj 相对 hand 的位姿 (pos + quat)，写入 eq_data[3:10]
        pos_h, quat_h = self.data.xpos[bh], self.data.xquat[bh]
        pos_o, quat_o = self.data.xpos[bo], self.data.xquat[bo]
        neg_h = np.zeros(4)
        mujoco.mju_negQuat(neg_h, quat_h)
        relpos = np.zeros(3)
        mujoco.mju_rotVecQuat(relpos, pos_o - pos_h, neg_h)
        relquat = np.zeros(4)
        mujoco.mju_mulQuat(relquat, neg_h, quat_o)
        self.model.eq_data[eid, 0:3] = 0.0          # anchor (body1 系)，用 relpose 即可
        self.model.eq_data[eid, 3:6] = relpos
        self.model.eq_data[eid, 6:10] = relquat
        self.model.eq_data[eid, 10] = 1.0           # torquescale
        self.data.eq_active[eid] = 1

    def detach_object(self) -> None:
        eid = self._weld_id()
        if eid >= 0:
            self.data.eq_active[eid] = 0

    def is_object_attached(self) -> bool:
        eid = self._weld_id()
        return eid >= 0 and bool(self.data.eq_active[eid])

    def gripper_width(self) -> float:
        f1 = self.model.jnt_qposadr[self._jid("finger_joint1")]
        f2 = self.model.jnt_qposadr[self._jid("finger_joint2")]
        return float(self.data.qpos[f1] + self.data.qpos[f2])

    # ----------------------------------------------------------------- 位姿查询（世界系 4x4）
    def body_pose(self, name: str) -> np.ndarray:
        bid = self._bid(name)
        R = self.data.xmat[bid].reshape(3, 3)
        p = self.data.xpos[bid]
        return make_transform(R, p)

    def site_pose(self, name: str) -> np.ndarray:
        sid = self._sid(name)
        R = self.data.site_xmat[sid].reshape(3, 3)
        p = self.data.site_xpos[sid]
        return make_transform(R, p)

    def ee_pose(self) -> np.ndarray:
        return self.site_pose(self.ee_site)

    def body_position(self, name: str) -> np.ndarray:
        return self.data.xpos[self._bid(name)].copy()

    def camera_pose(self, name: str) -> np.ndarray:
        """相机在世界系的位姿（MuJoCo 相机 z 轴朝外，看向 -z）。"""
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        R = self.data.cam_xmat[cid].reshape(3, 3)
        p = self.data.cam_xpos[cid]
        return make_transform(R, p)

    def camera_fovy(self, name: str) -> float:
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        return float(self.model.cam_fovy[cid])

    @property
    def time(self) -> float:
        return float(self.data.time)
