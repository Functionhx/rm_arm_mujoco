"""视觉引导取放任务有限状态机。

状态迁移与报告流程图一致：
SEARCH -> APPROACH -> DESCEND -> GRASP -> LIFT -> MOVE -> INSERT -> RELEASE -> RETREAT -> DONE
每个状态调用规划、逆运动学与控制模块，并根据反馈判断迁移；失败设有限次重试。
"""
from __future__ import annotations

from enum import Enum, auto

import numpy as np

from ..planning.grasp_planner import GraspPlanner
from ..utils.logger import get_logger

log = get_logger("fsm")


class State(Enum):
    SEARCH = auto()
    APPROACH = auto()
    DESCEND = auto()
    GRASP = auto()
    LIFT = auto()
    MOVE = auto()
    INSERT = auto()
    RELEASE = auto()
    RETREAT = auto()
    DONE = auto()
    FAILED = auto()


class PickPlaceStateMachine:
    def __init__(self, env, arm, gripper, pose_estimator, planner: GraspPlanner,
                 task_cfg: dict, recorder=None):
        self.env = env
        self.arm = arm
        self.gripper = gripper
        self.pose_estimator = pose_estimator
        self.planner = planner
        self.cfg = task_cfg
        self.recorder = recorder
        self.object_body = task_cfg["task"]["object_body"]
        self.place_body = task_cfg["task"]["place_body"]
        self.max_retries = task_cfg["state_machine"]["max_detect_retries"]
        self.state = State.SEARCH
        self.history: list[str] = []
        self.plan = None
        self.est_error = None

    def _render_obs(self):
        rgb, depth = self.pose_estimator.camera.render()
        return rgb, depth

    def run(self) -> dict:
        """执行完整取放流程，返回结果字典。"""
        retries = 0
        while self.state not in (State.DONE, State.FAILED):
            self.history.append(self.state.name)
            if self.recorder is not None:
                self.recorder.on_state(self.state.name)

            if self.state == State.SEARCH:
                rgb, depth = self._render_obs()
                pose = self.pose_estimator.estimate(rgb, depth)
                if not pose.found:
                    retries += 1
                    log.warning("检测失败，重试 %d/%d", retries, self.max_retries)
                    if retries > self.max_retries:
                        self.state = State.FAILED
                        continue
                    self.env.step(50)
                    continue
                # 记录视觉估计误差（相对仿真真值）
                gt = self.env.body_position(self.object_body)
                self.est_error = float(np.linalg.norm(pose.position_world[:2] - gt[:2]))
                obj_xy = pose.position_world[:2]
                place_xy = self.env.body_position(self.place_body)[:2]
                self.plan = self.planner.plan(
                    obj_xy, place_xy,
                    object_base_z=self.planner.table_top,
                    place_base_z=self.env.body_position(self.place_body)[2],
                )
                log.info("检测成功: 目标 xy=(%.3f,%.3f) 视觉误差=%.1fmm",
                         obj_xy[0], obj_xy[1], self.est_error * 1000)
                self.state = State.APPROACH

            elif self.state == State.APPROACH:
                self.gripper.open(recorder=self.recorder)
                self.arm.go_pose(self.plan.pre_grasp, num=70, recorder=self.recorder)
                self.state = State.DESCEND

            elif self.state == State.DESCEND:
                self.arm.move_cartesian(self.plan.grasp, num=45, hold=40, recorder=self.recorder)
                self.state = State.GRASP

            elif self.state == State.GRASP:
                ok = self.gripper.close(steps=90, recorder=self.recorder,
                                        attach_body=self.object_body)
                if not ok:
                    log.warning("抓取失败，回到 SEARCH")
                    self.gripper.open(recorder=self.recorder)
                    self.state = State.SEARCH
                    continue
                self.state = State.LIFT

            elif self.state == State.LIFT:
                self.arm.move_cartesian(self.plan.lift, num=45, hold=15, recorder=self.recorder)
                self.state = State.MOVE

            elif self.state == State.MOVE:
                self.arm.go_pose(self.plan.place_above, num=80, recorder=self.recorder)
                self.state = State.INSERT

            elif self.state == State.INSERT:
                self.arm.move_cartesian(self.plan.place, num=45, hold=25, recorder=self.recorder)
                self.state = State.RELEASE

            elif self.state == State.RELEASE:
                self.gripper.open(steps=60, recorder=self.recorder)
                self.env.step(150)
                self.state = State.RETREAT

            elif self.state == State.RETREAT:
                self.arm.move_cartesian(self.plan.retreat, num=40, hold=15, recorder=self.recorder)
                self.state = State.DONE

        return self._result()

    def _result(self) -> dict:
        obj = self.env.body_position(self.object_body)
        place = self.env.body_position(self.place_body)
        place_xy_err = float(np.linalg.norm(obj[:2] - place[:2]))
        success = (self.state == State.DONE) and (place_xy_err < 0.08)
        return {
            "success": success,
            "final_state": self.state.name,
            "place_xy_error": place_xy_err,
            "vision_error": self.est_error,
            "object_pos": obj.tolist(),
            "place_pos": place.tolist(),
            "history": self.history,
        }
