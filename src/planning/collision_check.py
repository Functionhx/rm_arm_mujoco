"""简化碰撞检查（可选）。

用 MuJoCo 的碰撞检测统计非预期接触；抓取阶段与目标物体的接触为预期，予以排除。
"""
from __future__ import annotations

import mujoco


def count_unexpected_contacts(env, allow_bodies=("energy_unit",)) -> int:
    """统计机械臂与非允许物体间的接触数量。"""
    d = env.data
    m = env.model
    allow_ids = set()
    for name in allow_bodies:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            allow_ids.add(bid)
    n = 0
    for i in range(d.ncon):
        c = d.contact[i]
        b1 = m.geom_bodyid[c.geom1]
        b2 = m.geom_bodyid[c.geom2]
        if b1 in allow_ids or b2 in allow_ids:
            continue
        n += 1
    return n
