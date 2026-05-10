"""
任务配置注册表。原版本一次性 import 所有真机 + 仿真 task config，
任何一个 task 的依赖（如 pyrealsense2 / easyhid / pynput）缺失都会让 import 链崩塌。
改为：每个 task 单独 try/except，缺依赖就跳过并打 warning，让 sim-only 用户也能跑。
"""
import importlib
import warnings

CONFIG_MAPPING = {}

_TASKS = [
    ("ram_insertion",        "experiments.ram_insertion.config"),
    ("usb_pickup_insertion", "experiments.usb_pickup_insertion.config"),
    ("object_handover",      "experiments.object_handover.config"),
    ("egg_flip",             "experiments.egg_flip.config"),
    ("pick_cube_sim",        "experiments.pick_cube_sim.config"),
]

for _task_name, _mod_path in _TASKS:
    try:
        _mod = importlib.import_module(_mod_path)
        CONFIG_MAPPING[_task_name] = _mod.TrainConfig
    except Exception as _e:
        warnings.warn(
            f"[mappings] Skip task '{_task_name}': {type(_e).__name__}: {_e}"
        )