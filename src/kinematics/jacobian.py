"""MuJoCo 雅可比封装：末端站点相对手臂关节的 6xN 雅可比。"""
from __future__ import annotations

import mujoco
import numpy as np


def site_jacobian(model, data, site_id: int, dof_adr: np.ndarray) -> np.ndarray:
    """返回末端站点关于指定关节自由度的 6xN 雅可比。

    上 3 行为平移雅可比 Jp，下 3 行为旋转雅可比 Jr（世界系）。
    dof_adr 为手臂各关节在 qvel 中的地址（本机每关节 1 个自由度）。
    """
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    Jp = jacp[:, dof_adr]
    Jr = jacr[:, dof_adr]
    return np.vstack([Jp, Jr])
