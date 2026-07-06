"""取放任务主逻辑：组装各模块，运行状态机，返回结果。"""
from __future__ import annotations

import numpy as np

from ..control.arm_controller import ArmController
from ..control.gripper_controller import GripperController
from ..geometry.frames import FrameManager
from ..kinematics.inverse_kinematics import IKSolver
from ..sim.camera import RGBDCamera
from ..sim.mujoco_env import MujocoEnv, load_yaml
from ..utils.logger import get_logger
from ..vision.pose_estimator import PoseEstimator
from ..planning.grasp_planner import GraspPlanner
from .state_machine import PickPlaceStateMachine

log = get_logger("task")


class PickPlaceTask:
    def __init__(self, env: MujocoEnv | None = None, recorder=None):
        self.env = env or MujocoEnv()
        self.task_cfg = load_yaml("configs/task.yaml")
        self.sim_cfg = self.env.sim_cfg
        cam_cfg = self.sim_cfg["camera"]
        self.camera = RGBDCamera(self.env, cam_cfg["name"], cam_cfg["width"], cam_cfg["height"])
        self.frames = FrameManager(self.env, base_body=self.env.base_body,
                                   camera_name=cam_cfg["name"])
        self.pose_estimator = PoseEstimator(self.camera, self.frames,
                                            self.task_cfg["vision"], table_top=0.40)
        self.ik = IKSolver(self.env)
        self.arm = ArmController(self.env, self.ik)
        self.gripper = GripperController(self.env)
        self.planner = GraspPlanner(self.task_cfg, table_top=0.40)
        self.recorder = recorder

    def randomize_object(self, rng: np.random.Generator) -> None:
        """随机化能量单元初始水平位置（在工作台可达范围内）。"""
        x = 0.46 + rng.uniform(-0.05, 0.06)
        y = 0.02 + rng.uniform(-0.10, 0.10)
        adr = self.env.model.jnt_qposadr[self.env._jid("energy_unit_free")]
        self.env.data.qpos[adr:adr + 3] = [x, y, 0.4005]
        self.env.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        self.env.data.qvel[:] = 0
        import mujoco
        mujoco.mj_forward(self.env.model, self.env.data)
        self.env.step(120)

    def run(self, recorder=None) -> dict:
        rec = recorder or self.recorder
        fsm = PickPlaceStateMachine(
            self.env, self.arm, self.gripper, self.pose_estimator,
            self.planner, self.task_cfg, recorder=rec,
        )
        result = fsm.run()
        log.info("任务结束: success=%s state=%s place_err=%.1fmm",
                 result["success"], result["final_state"], result["place_xy_error"] * 1000)
        return result
