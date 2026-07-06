"""抓取规划：由目标位姿与放置目标生成 pre-grasp / grasp / lift / place 位姿序列。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.transforms import make_transform, rpy_to_matrix


# 自上而下抓取姿态：TCP 的 z 轴指向 -world z（竖直向下），夹爪沿水平方向闭合
R_TOPDOWN = rpy_to_matrix(np.pi, 0.0, 0.0)


@dataclass
class GraspPlan:
    pre_grasp: np.ndarray
    grasp: np.ndarray
    lift: np.ndarray
    place_above: np.ndarray
    place: np.ndarray
    retreat: np.ndarray


class GraspPlanner:
    def __init__(self, task_cfg: dict, table_top: float = 0.40):
        g = task_cfg["grasp"]
        p = task_cfg["place"]
        self.approach_height = g["approach_height"]
        self.grasp_z_offset = g["grasp_z_offset"]
        self.lift_height = g["lift_height"]
        self.place_above_height = p["above_height"]
        self.insert_depth = p["insert_depth"]
        self.table_top = table_top

    def _pose(self, xyz: np.ndarray) -> np.ndarray:
        return make_transform(R_TOPDOWN, np.asarray(xyz, float))

    def plan(self, object_xy: np.ndarray, place_xy: np.ndarray,
             object_base_z: float | None = None,
             place_base_z: float | None = None) -> GraspPlan:
        """object_xy: 目标水平位置 (x,y)；place_xy: 放置目标水平位置。

        z 由桌面/放置面先验确定，保证插入-放置的稳定性。
        """
        obj_z = self.table_top if object_base_z is None else object_base_z
        plc_z = self.table_top if place_base_z is None else place_base_z

        grasp_pt = np.array([object_xy[0], object_xy[1], obj_z + self.grasp_z_offset])
        pre = grasp_pt + np.array([0, 0, self.approach_height])
        lift = grasp_pt + np.array([0, 0, self.lift_height])

        place_pt = np.array([place_xy[0], place_xy[1], plc_z + self.insert_depth])
        place_above = place_pt + np.array([0, 0, self.place_above_height])
        retreat = place_above + np.array([0, 0, 0.06])

        return GraspPlan(
            pre_grasp=self._pose(pre),
            grasp=self._pose(grasp_pt),
            lift=self._pose(lift),
            place_above=self._pose(place_above),
            place=self._pose(place_pt),
            retreat=self._pose(retreat),
        )
