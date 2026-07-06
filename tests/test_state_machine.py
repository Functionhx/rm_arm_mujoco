"""状态机集成测试：完整取放任务应成功并把物体放入放置区。"""
import numpy as np

from src.task.pick_place_task import PickPlaceTask


def test_pick_place_success():
    task = PickPlaceTask()
    task.env.reset()
    result = task.run()
    assert result["final_state"] == "DONE"
    assert result["success"]
    assert result["place_xy_error"] < 0.08


def test_state_sequence_reaches_grasp():
    task = PickPlaceTask()
    task.env.reset()
    result = task.run()
    # 关键状态都应出现在迁移历史中
    for s in ["SEARCH", "APPROACH", "DESCEND", "GRASP", "LIFT", "MOVE", "INSERT", "RELEASE"]:
        assert s in result["history"]
