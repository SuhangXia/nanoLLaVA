#!/usr/bin/env python3
"""
VLA PyBullet 评估：支持 Franka Panda 或 WidowX 250，按 Bridge V2 “HD table & OTS” 做关键对齐。

- Robot: franka_panda (pybullet_data) 或 widowx_250 (assets/widowx_250)。
- HD table: 启动时从 URDF 估计链长；若与标称 (0.155,0.155,0.110)m 偏差 >10%，用 globalScaling 缩放 URDF。
- World: base=[0,0,0] 面向 +X；红方块 size=0.02m，pos=[0.22,0,0.02]；dist>0.28m 中止。
- OTS camera: eye=[-0.10,-0.15,0.40] target=[0.25,0,0.05] fov=75（RealSense D435 风格）。
- Action: WidowX xyz_scale=0.02；Z 最小 clip=0.03m；g>0 张开、g<0 闭合。
"""

import os
import sys
import argparse
import pickle
import time
import numpy as np
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from types import SimpleNamespace

os.environ.setdefault("NUMBA_DISABLE_JIT_CACHE", "1")

# PyTorch / transformers 兼容
if not hasattr(torch.utils._pytree, "register_pytree_node") and hasattr(torch.utils._pytree, "_register_pytree_node"):
    def _compat_rpn(ty, fn, uf, *a, serialized_type_name=None, **k):
        return torch.utils._pytree._register_pytree_node(ty, fn, uf)
    torch.utils._pytree.register_pytree_node = _compat_rpn

import transformers
from bunny.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
from bunny.model import BunnyQwenForCausalLM

# transformers cache 兼容：本仓库的 qwen2 实现会调用 past_key_values.get_usable_length，
# 但较新的 transformers 里 DynamicCache 对应方法名是 get_seq_length。
try:
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, "get_usable_length") and hasattr(DynamicCache, "get_seq_length"):
        DynamicCache.get_usable_length = DynamicCache.get_seq_length  # type: ignore[attr-defined]
except Exception:
    pass

# --- 默认路径 ---
DEFAULT_VLA_CKPT = "./outputs/vla_phase1_bf16"
DEFAULT_ACTION_HEAD = "action_head_step5500.bin"
DEFAULT_MODEL = "BAAI/Bunny-v1_0-2B-zh"
DEFAULT_VISION = "siglip-so400m-patch14-384"
DEFAULT_PROMPT = ""

# --- PyBullet 常量 ---
# Franka Panda
PANDA_ARM_JOINTS = list(range(7))
PANDA_FINGER_JOINTS = [9, 10]
PANDA_EE_LINK_INDEX = 6
# WidowX 250 (Black arm, assets/widowx_250/widowx_250.urdf)
WIDOWX_ARM_JOINTS = [0, 1, 2, 3, 4, 5]
WIDOWX_FINGER_JOINTS = [6, 7]
WIDOWX_EE_LINK_INDEX = 5
# Official OTS (Bridge V2) 相机 TF：相机在 base 后方略偏右，朝向工作区中心
CAM_EYE_BRIDGE = [-0.10, -0.15, 0.40]
CAM_TARGET_BRIDGE = [0.25, 0.0, 0.05]
CAM_FOV_BRIDGE = 75.0
# WidowX 初始关节 (IK 失败时回退)
WIDOWX_INIT_ARM = [0.0, -0.5, 1.0, -1.5, 0.0, 0.0]
# Warm-start Pose（step=0 强制）：不使用随机 IK
WIDOWX_WARMSTART_ARM = [0.0, -0.5, 0.8, -1.0, 0.0, 0.0]

# Interbotix / WidowX 250 近似标称（Bridge V2）
_WIDOWX_SPEC_LINKS_M = {"shoulder": 0.155, "elbow": 0.155, "tip": 0.110}
_WIDOWX_SPEC_LIMITS_RAD = {
    "joint_1": (-3.14, 3.14),
    "joint_2": (-1.88, 1.99),
    "joint_3": (-2.14, 1.60),
    "joint_4": (-1.74, 2.14),
    "joint_5": (-3.14, 3.14),
}


def _parse_urdf_joint_origins(urdf_path: str) -> dict:
    """
    读取 URDF 里各关节的 origin.xyz（以关节名为 key）。
    仅用于 kinematic/scale 自检（HD table）。
    """
    out = {}
    try:
        root = ET.parse(urdf_path).getroot()
        for j in root.findall("joint"):
            name = j.attrib.get("name", "")
            origin = j.find("origin")
            if origin is None:
                continue
            xyz_s = origin.attrib.get("xyz", "0 0 0")
            xyz = [float(x) for x in xyz_s.split()]
            out[name] = xyz
    except Exception:
        return {}
    return out


def _widowx_detected_link_lengths_from_urdf(urdf_path: str) -> dict:
    """
    从 URDF joint origins 粗略估计链长（单位 m）。
    注意：这是「我们当前 URDF」的几何近似，用于提示是否与 Interbotix 标准差异过大。
    """
    origins = _parse_urdf_joint_origins(urdf_path)
    # 本仓库 URDF: joint_3(y=0.04), joint_4(y=0.035), joint_5(y=0.02), joint_6(y=0.015), gripper_left(y=0.006,z=0.01)
    j1 = origins.get("joint_1", [0.0, 0.0, 0.0])
    j2 = origins.get("joint_2", [0.0, 0.0, 0.0])
    j3 = origins.get("joint_3", [0.0, 0.0, 0.0])
    j4 = origins.get("joint_4", [0.0, 0.0, 0.0])
    j5 = origins.get("joint_5", [0.0, 0.0, 0.0])
    j6 = origins.get("joint_6", [0.0, 0.0, 0.0])
    gl = origins.get("gripper_left", [0.0, 0.0, 0.0])
    shoulder = float(np.linalg.norm(np.asarray(j3, dtype=np.float32)))
    elbow = float(np.linalg.norm(np.asarray(j4, dtype=np.float32)))
    tip = float(
        np.linalg.norm(np.asarray(j5, dtype=np.float32))
        + np.linalg.norm(np.asarray(j6, dtype=np.float32))
        + np.linalg.norm(np.asarray(gl, dtype=np.float32))
    )
    return dict(
        shoulder=shoulder,
        elbow=elbow,
        tip=tip,
        urdf_joint_origins=origins,
    )


def _log_widowx_hd_table_warnings(p, robot_id: int, arm_joints: list, urdf_path: str, scale: float = 1.0):
    """
    检查 URDF 近似链长与关节限制是否与 Interbotix/WidowX 250 标称差异显著。
    仅打印 warning，不中止。
    """
    det = _widowx_detected_link_lengths_from_urdf(urdf_path)
    if det:
        # 仅告警（缩放逻辑在 loadURDF 前执行）；阈值：>10%
        for k, v_ref in _WIDOWX_SPEC_LINKS_M.items():
            v = float(det.get(k, 0.0)) * float(scale)
            if v_ref > 0 and v > 0 and abs(v - v_ref) / v_ref > 0.10:
                print(f"[warn] WidowX URDF link length mismatch: {k}={v:.3f}m vs spec~{v_ref:.3f}m (>10%)")
    # 关节限制（来自 PyBullet jointInfo）
    name_map = {
        0: "joint_1",
        1: "joint_2",
        2: "joint_3",
        3: "joint_4",
        4: "joint_5",
    }
    for jid in arm_joints[:5]:
        info = p.getJointInfo(robot_id, jid)
        lo, hi = float(info[8]), float(info[9])
        jname = name_map.get(jid, f"joint_{jid+1}")
        if jname in _WIDOWX_SPEC_LIMITS_RAD:
            slo, shi = _WIDOWX_SPEC_LIMITS_RAD[jname]
            if abs(lo - slo) > 0.25 or abs(hi - shi) > 0.25:
                print(f"[warn] WidowX URDF joint limit mismatch: {jname}=[{lo:.2f},{hi:.2f}] vs spec~[{slo:.2f},{shi:.2f}]")


def _get_arm_joint_limits(p, robot_id, joint_ids):
    lo, hi, ranges = [], [], []
    for j in joint_ids:
        info = p.getJointInfo(robot_id, j)
        lo_j, hi_j = float(info[8]), float(info[9])
        # 有些 revolute joint 的 limit 可能是 (0, -1) 这种占位；做个兜底
        if hi_j <= lo_j:
            lo_j, hi_j = -2.9, 2.9
        lo.append(lo_j)
        hi.append(hi_j)
        ranges.append(hi_j - lo_j)
    return lo, hi, ranges


def _build_ik_params(p, robot_id, arm_joints):
    """给 IK 加 joint limits + rest poses。不传 jointDamping，避免与 PyBullet 内部 DoF 数不一致时的刷屏警告。"""
    lo, hi, jr = _get_arm_joint_limits(p, robot_id, arm_joints)
    rest = [float(p.getJointState(robot_id, j)[0]) for j in arm_joints]
    return dict(lowerLimits=lo, upperLimits=hi, jointRanges=jr, restPoses=rest)


def load_model_and_stats(vla_ckpt: str, action_head_file: str, model_name: str, vision_tower: str):
    """BF16 加载 Bunny-Qwen 与 action_head、action_mean_std.pkl。"""
    kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto", "low_cpu_mem_usage": True}
    model = BunnyQwenForCausalLM.from_pretrained(model_name, **kwargs)
    # 评估只需要 action head，不需要 KV cache；关掉可省内存并规避 cache 兼容问题
    try:
        model.config.use_cache = False
    except Exception:
        pass
    model.get_model().initialize_vision_modules(SimpleNamespace(
        vision_tower=vision_tower, pretrain_mm_mlp_adapter=None, mm_projector_type="mlp2x_gelu",
    ))
    vt = model.get_vision_tower()
    if not vt.is_loaded:
        vt.load_model()
    vt.to(device=torch.device("cuda"), dtype=torch.bfloat16)

    ah_path = os.path.join(vla_ckpt, action_head_file)
    if not os.path.isfile(ah_path):
        ah_alt = os.path.join(vla_ckpt, "action_head.bin")
        ah_path = ah_alt if os.path.isfile(ah_alt) else ah_path
    if not os.path.isfile(ah_path):
        raise FileNotFoundError(f"Not found: {ah_path} or action_head.bin")
    model.action_head.load_state_dict(torch.load(ah_path, map_location="cpu"))
    # 权重校验：确认 Step 5500 已加载、非全零
    w = model.action_head.mlp[0].weight
    print(f"[eval] action_head.mlp[0].weight mean: {w.float().mean().item():.6f}")
    model.eval()

    with open(os.path.join(vla_ckpt, "action_mean_std.pkl"), "rb") as f:
        stats = pickle.load(f)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return model, mean, std


def preprocess_image(arr: np.ndarray, processor, size=(384, 384)) -> torch.Tensor:
    """(H,W,3) [0,255] 或 [0,1] -> 384×384 -> processor -> pixel_values"""
    if arr.max() <= 1.0:
        arr = (np.clip(arr, 0, 1) * 255.0).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr).convert("RGB").resize(size, Image.BILINEAR)
    return processor(im, return_tensors="pt")["pixel_values"]


def predict(model, pixel_values, num_image_tokens, tokenizer, device, prompt: str = None):
    """Image-only 预测 7D 动作；prompt 暂未接入 action head，仅占位。"""
    B = pixel_values.shape[0]
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    input_ids = torch.tensor([[IMAGE_TOKEN_INDEX, eos]] * B, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
    labels = torch.full_like(input_ids, IGNORE_INDEX, device=device)
    with torch.no_grad():
        # 关键：mm_projector / vision tower 多为 BF16，这里必须把 images cast 到同 dtype
        img = pixel_values.to(device=device, dtype=next(model.parameters()).dtype)
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            images=img,
            action_labels=None,
            num_image_tokens=num_image_tokens,
            use_cache=False,
        )
    return out.action_prediction.cpu().float().numpy()


def denormalize(act_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return act_norm * std + mean


def _euler_from_quat(q: np.ndarray) -> np.ndarray:
    """[x,y,z,w] -> (roll, pitch, yaw) in radians."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1, 1)
    pitch = np.arcsin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float32)


def _quat_from_euler(rpy: np.ndarray) -> np.ndarray:
    """(roll, pitch, yaw) -> [x,y,z,w]."""
    r, p, y = rpy[0], rpy[1], rpy[2]
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([x, y, z, w], dtype=np.float32)


def action_7d_to_ik_targets(
    act_raw: np.ndarray,
    ee_pos: np.ndarray,
    ee_quat: np.ndarray,
    xyz_scale: float = 0.01,
    ori_scale: float = 0.1,
    ori_mode: str = "keep",
    fixed_rpy: tuple = (np.pi, 0.0, 0.0),
    workspace_low: tuple = (0.15, -0.45, 0.05),
    workspace_high: tuple = (0.80, 0.45, 0.80),
    min_z: float = 0.30,
    finger_max: float = 0.04,
) -> tuple:
    """
    [dx, dy, dz, dr, dp, dyaw, g] -> (target_pos, target_quat, finger_width, target_pos_before_clip).
    act_raw: denorm 后的 7D，clip 到 [-1,1] 后缩放。target_pos_before_clip 为 workspace clip 前的 ee_pos+dxyz。
    """
    act = np.asarray(act_raw, dtype=np.float32).reshape(-1)[:7]
    dxyz = np.clip(act[0:3], -1.0, 1.0) * xyz_scale
    target_pos_before_clip = np.asarray(ee_pos + dxyz, dtype=np.float32)
    target_pos = np.asarray(target_pos_before_clip, dtype=np.float32)
    # 工作空间夹紧 + 防撞桌（min_z）
    target_pos = np.clip(target_pos, np.array(workspace_low, dtype=np.float32), np.array(workspace_high, dtype=np.float32))
    target_pos[2] = max(float(target_pos[2]), float(min_z))

    if ori_mode == "delta":
        rpy = _euler_from_quat(ee_quat)
        drpy = np.clip(act[3:6], -1.0, 1.0) * ori_scale
        target_quat = _quat_from_euler(rpy + drpy)
    elif ori_mode == "fixed":
        target_quat = _quat_from_euler(np.array(fixed_rpy, dtype=np.float32))
    else:
        # keep：忽略 dr/dp/dyaw，保持当前朝向，避免夹爪“朝向 base”之类的漂移
        target_quat = np.asarray(ee_quat, dtype=np.float32)

    g = np.clip(act[6], -1.0, 1.0)
    # g>0 -> open (finger large), g<0 -> close (finger small). WidowX/PyBullet: finger_max=open, 0=closed.
    finger = finger_max * (g + 1.0) / 2.0
    return target_pos, target_quat, finger, target_pos_before_clip


def _make_colored_geom(p, geom, half_extents=None, radius=None, length=None, rgba=(1, 0, 0, 1), mass=0.05, pos=(0, 0, 0.3)):
    if geom == p.GEOM_BOX:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(half_extents))
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=list(half_extents), rgbaColor=list(rgba))
    elif geom == p.GEOM_CYLINDER:
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=float(radius), height=float(length))
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=float(radius), length=float(length), rgbaColor=list(rgba))
    elif geom == p.GEOM_SPHERE:
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=float(radius))
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=float(radius), rgbaColor=list(rgba))
    else:
        raise ValueError(f"Unsupported geom: {geom}")
    return p.createMultiBody(float(mass), col, vis, list(pos))


def build_scene(p, pybullet_data_path: str, scene: str = "cube", seed: int = 0, robot: str = "panda"):
    """地板、桌子、目标物体。robot=widowx 时桌子更小更近以配合 250mm 臂展。"""
    p.setGravity(0, 0, -10)
    plane_path = os.path.join(pybullet_data_path, "plane.urdf")
    if os.path.isfile(plane_path):
        p.loadURDF(plane_path)
    else:
        col = p.createCollisionShape(p.GEOM_PLANE)
        p.createMultiBody(0, col, -1, [0, 0, 0])

    if (robot or "panda").lower() == "widowx":
        # “HD table”：把桌面顶面固定在 z=0.01（world），地面 z=0；base 在 [0,0,0]
        # 这样 cube center z=0.02（2cm 物体）时，底面刚好落在桌面上 (0.02-0.01=0.01)
        t_pos, t_he = [0.30, 0.0, -0.005], [0.25, 0.20, 0.015]
    else:
        t_pos, t_he = [0.5, 0, 0.25], [0.3, 0.25, 0.02]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=t_he)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=t_he, rgbaColor=[0.6, 0.5, 0.4, 1])
    table_id = p.createMultiBody(0, col, vis, t_pos)
    table_top_z = float(t_pos[2]) + float(t_he[2])

    rng = np.random.RandomState(int(seed))
    objs = []
    def _pos(x, y, z_off=0.04):
        return (float(x), float(y), float(table_top_z + z_off))

    scene = (scene or "cube").lower()
    _x0 = 0.22 if (robot or "").lower() == "widowx" else 0.50
    _y0 = 0.0
    if scene == "cube":
        if (robot or "").lower() == "widowx":
            # Red cube: size=0.02m (half=0.01), pos=[0.22,0,0.02]
            oid = _make_colored_geom(p, p.GEOM_BOX, half_extents=(0.01, 0.01, 0.01), rgba=(1, 0, 0, 1), pos=(0.22, 0.0, 0.02))
        else:
            oid = _make_colored_geom(p, p.GEOM_BOX, half_extents=(0.02, 0.02, 0.02), rgba=(1, 0, 0, 1), pos=_pos(_x0, _y0, 0.04))
        objs.append((oid, "red cube"))
    elif scene == "cylinder":
        oid = _make_colored_geom(p, p.GEOM_CYLINDER, radius=0.02, length=0.10, rgba=(0.2, 0.4, 1, 1), pos=_pos(_x0 + 0.02, _y0 + 0.05, 0.06))
        objs.append((oid, "blue cylinder"))
    elif scene == "sphere":
        oid = _make_colored_geom(p, p.GEOM_SPHERE, radius=0.025, rgba=(0.2, 1, 0.2, 1), pos=_pos(_x0 - 0.02, _y0 - 0.06, 0.05))
        objs.append((oid, "green sphere"))
    elif scene == "multi":
        oid1 = _make_colored_geom(p, p.GEOM_BOX, half_extents=(0.02, 0.02, 0.02), rgba=(1, 0, 0, 1), pos=_pos(_x0, _y0 - 0.05, 0.04))
        oid2 = _make_colored_geom(p, p.GEOM_CYLINDER, radius=0.02, length=0.10, rgba=(0.2, 0.4, 1, 1), pos=_pos(_x0 + 0.05, _y0 + 0.06, 0.06))
        objs.extend([(oid1, "red cube"), (oid2, "blue cylinder")])
    elif scene == "random":
        x_lo, x_hi = (0.12, 0.28) if (robot or "").lower() == "widowx" else (0.40, 0.62)
        y_lo, y_hi = (-0.10, 0.10) if (robot or "").lower() == "widowx" else (-0.16, 0.16)
        n = int(rng.randint(1, 4))
        for k in range(n):
            kind = rng.choice(["box", "cyl", "sph"])
            x = rng.uniform(x_lo, x_hi)
            y = rng.uniform(y_lo, y_hi)
            if kind == "box":
                hs = (0.015 + rng.rand() * 0.015, 0.015 + rng.rand() * 0.015, 0.015 + rng.rand() * 0.02)
                oid = _make_colored_geom(p, p.GEOM_BOX, half_extents=hs, rgba=tuple(rng.rand(3).tolist() + [1.0]), pos=_pos(x, y, 0.03 + hs[2]))
                objs.append((oid, f"box{k}"))
            elif kind == "cyl":
                r = 0.015 + rng.rand() * 0.015
                L = 0.05 + rng.rand() * 0.10
                oid = _make_colored_geom(p, p.GEOM_CYLINDER, radius=r, length=L, rgba=tuple(rng.rand(3).tolist() + [1.0]), pos=_pos(x, y, 0.03 + L / 2))
                objs.append((oid, f"cyl{k}"))
            else:
                r = 0.018 + rng.rand() * 0.02
                oid = _make_colored_geom(p, p.GEOM_SPHERE, radius=r, rgba=tuple(rng.rand(3).tolist() + [1.0]), pos=_pos(x, y, 0.03 + r))
                objs.append((oid, f"sph{k}"))
    else:
        raise ValueError(f"Unknown scene: {scene}")

    return dict(table_id=table_id, table_top_z=table_top_z, objects=objs)


def parse_args():
    p = argparse.ArgumentParser(description="VLA PyBullet 评估：WidowX 250 (Bridge V2) 或 Franka Panda，384 相机，IK，输出 mp4。")
    p.add_argument("--robot", type=str, default="widowx", choices=["panda", "widowx"], help="panda=pybullet_data/franka_panda, widowx=assets/widowx_250 (Black arm, Bridge V2)")
    p.add_argument("--vla_ckpt", type=str, default=DEFAULT_VLA_CKPT)
    p.add_argument("--action_head", type=str, default=DEFAULT_ACTION_HEAD)
    p.add_argument("--model_name_or_path", type=str, default=DEFAULT_MODEL)
    p.add_argument("--vision_tower", type=str, default=DEFAULT_VISION)
    p.add_argument("--scene", type=str, default="cube", choices=["cube", "cylinder", "sphere", "multi", "random"])
    p.add_argument("--seed", type=int, default=0, help="random 场景的随机种子")
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="文本提示（空则按 scene 自动生成）")
    p.add_argument("--steps", type=int, default=600, help="推理步数")
    p.add_argument("--output_mp4", type=str, default="vla_eval_pybullet.mp4")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--cam_size", type=int, default=384, help="getCameraImage 宽高")
    p.add_argument("--xyz_scale", type=float, default=0.02, help="dx/dy/dz 每步的米尺度；WidowX 建议 0.02（更贴近 Bridge 步幅）")
    p.add_argument("--ori_scale", type=float, default=0.10, help="dr/dp/dyaw 每步的弧度尺度（ori_mode=delta 时生效）")
    p.add_argument("--ori_mode", type=str, default="keep", choices=["keep", "delta", "fixed"], help="末端朝向模式：keep(默认更稳)/delta/ fixed(夹爪朝下)")
    p.add_argument("--gui", action="store_true", help="强制 p.GUI；否则 DISPLAY 时 GUI，无则 DIRECT")
    p.add_argument("--direct", action="store_true", help="强制 p.DIRECT 离屏")
    p.add_argument("--tiny_renderer", action="store_true", help="强制使用 ER_TINY_RENDERER（不依赖 OpenGL，最稳但更慢）")
    return p.parse_args()


def main():
    args = parse_args()

    import pybullet as p
    import pybullet_data
    import imageio

    use_gui = args.gui or (not args.direct and os.environ.get("DISPLAY"))
    mode = p.GUI if use_gui else p.DIRECT
    cid = p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot = (args.robot or "widowx").lower()

    if robot == "widowx":
        urdf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "widowx_250", "widowx_250.urdf")
        if not os.path.isfile(urdf):
            raise FileNotFoundError(f"WidowX 250 URDF not found: {urdf}")
        ARM_JOINTS, FINGER_JOINTS = WIDOWX_ARM_JOINTS, WIDOWX_FINGER_JOINTS
        EE_LINK_INDEX, FINGER_MAX = WIDOWX_EE_LINK_INDEX, 0.02
        WS_LOW, WS_HI = (0.05, -0.25, 0.03), (0.30, 0.25, 0.25)  # 更保守：Z>=0.03 防撞桌
        # HD table audit: 若 URDF 链长与标称偏差 >10%，用 globalScaling 缩放
        det0 = _widowx_detected_link_lengths_from_urdf(urdf)
        widowx_scale = 1.0
        if det0:
            # 以 shoulder/elbow 两段为主做缩放（tip 段在最小 URDF 上不稳定）
            ratios = []
            for k in ("shoulder", "elbow"):
                v_ref = float(_WIDOWX_SPEC_LINKS_M[k])
                v = float(det0.get(k, 0.0))
                if v > 1e-6:
                    ratios.append(v_ref / v)
            need_scale = any(
                (float(det0.get(k, 0.0)) > 1e-6 and abs(float(det0.get(k, 0.0)) - float(_WIDOWX_SPEC_LINKS_M[k])) / float(_WIDOWX_SPEC_LINKS_M[k]) > 0.10)
                for k in ("shoulder", "elbow")
            )
            if need_scale and ratios:
                widowx_scale = float(np.mean(np.asarray(ratios, dtype=np.float32)))
                print(f"[warn] HD table: applying globalScaling={widowx_scale:.3f} to match shoulder/elbow (spec: 0.155/0.155m)")
    else:
        urdf = os.path.join(pybullet_data.getDataPath(), "franka_panda", "panda.urdf")
        ARM_JOINTS, FINGER_JOINTS = PANDA_ARM_JOINTS, PANDA_FINGER_JOINTS
        EE_LINK_INDEX, FINGER_MAX = PANDA_EE_LINK_INDEX, 0.04
        WS_LOW, WS_HI = (0.15, -0.45, 0.05), (0.80, 0.45, 0.80)

    # 先建场景以得到 table_top_z；WidowX 的 base 需放在桌面上
    scene_info = build_scene(p, pybullet_data.getDataPath(), scene=args.scene, seed=args.seed, robot=robot)
    table_top_z = float(scene_info["table_top_z"])
    # Z safety：WidowX 最小 z=0.03m（绝对世界坐标）
    min_z = 0.03 if robot == "widowx" else (table_top_z + 0.03)

    if robot == "widowx":
        # 180° 修正：base 固定在世界原点，朝 +X（URDF 默认朝向）
        base_pos = [0.0, 0.0, 0.0]
        robot_id = p.loadURDF(urdf, basePosition=base_pos, useFixedBase=True, globalScaling=float(widowx_scale))
    else:
        robot_id = p.loadURDF(urdf, useFixedBase=True)

    for j in ARM_JOINTS + FINGER_JOINTS:
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, force=200)

    ik_params = _build_ik_params(p, robot_id, ARM_JOINTS)
    widowx_hover_arm = None
    if robot == "widowx":
        widowx_hover_arm = list(WIDOWX_WARMSTART_ARM)
        # HD Table (kinematic) 自检：URDF 与 Interbotix 标称对比（只告警）
        _log_widowx_hd_table_warnings(p, robot_id, ARM_JOINTS, urdf_path=urdf, scale=float(widowx_scale))
        for i, j in enumerate(ARM_JOINTS):
            p.resetJointState(robot_id, j, widowx_hover_arm[i])
        for j in FINGER_JOINTS:
            p.resetJointState(robot_id, j, 0.01)  # 夹爪微开

    if use_gui:
        if robot == "widowx":
            p.resetDebugVisualizerCamera(cameraDistance=0.95, cameraYaw=45, cameraPitch=-40, cameraTargetPosition=list(CAM_TARGET_BRIDGE))
        else:
            p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=60, cameraPitch=-30, cameraTargetPosition=[0.4, 0, 0.35])
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 0)
    for _ in range(30):
        p.stepSimulation()
        if use_gui:
            time.sleep(1.0 / 60.0)

    # Pre-flight Reachability：dist(cube, base) > 0.28m 则中止
    if robot == "widowx" and (args.scene or "cube").lower() == "cube" and scene_info.get("objects"):
        cube_id = scene_info["objects"][0][0]
        pos, _ = p.getBasePositionAndOrientation(cube_id)
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        dist = np.sqrt(x * x + y * y)
        if dist > 0.28:
            raise RuntimeError(f"Target out of WidowX reach: dist={dist:.4f} (>0.28)")

    # 相机：WidowX 用 Bridge V2 肩后；投影 1:1、nearVal=0.01 与 384×384 resize 一致，避免近处裁切
    if robot == "widowx":
        view = p.computeViewMatrix(CAM_EYE_BRIDGE, CAM_TARGET_BRIDGE, [0.0, 0.0, 1.0])
        proj = p.computeProjectionMatrixFOV(CAM_FOV_BRIDGE, 1.0, 0.01, 10.0)  # aspect=1:1, nearVal=0.01
    else:
        view = p.computeViewMatrixFromYawPitchRoll([0.4, 0, 0.35], 1.0, 60, -25, 0, 2)
        proj = p.computeProjectionMatrixFOV(60, 1.0, 0.01, 10)

    def get_rgb():
        w, h = args.cam_size, args.cam_size
        renderer = p.ER_TINY_RENDERER if (args.tiny_renderer or args.direct or not use_gui) else p.ER_BULLET_HARDWARE_OPENGL
        _, _, rgba, _, _ = p.getCameraImage(w, h, view, proj, renderer=renderer)
        return np.asarray(rgba)[:, :, :3]  # RGB, [0,255]

    # VLA
    print("[eval] Loading VLA (BF16) and action_head ...")
    model, mean, std = load_model_and_stats(
        args.vla_ckpt, args.action_head, args.model_name_or_path, args.vision_tower,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    processor = model.get_vision_tower().image_processor
    num_image_tokens = model.get_vision_tower().num_patches
    device = next(model.parameters()).device
    if not args.prompt:
        prompt_map = {
            "cube": "Lift the red cube",
            "cylinder": "Pick up the blue cylinder",
            "sphere": "Grasp the green sphere",
            "multi": "Pick up the red cube",
            "random": "Pick up the object",
        }
        args.prompt = prompt_map.get(args.scene, "Pick up the object")
    print(f"[eval] Scene={args.scene} Prompt (placeholder): \"{args.prompt}\"")

    frames = []
    for step in range(args.steps):
        # 第一步前再次强制 WidowX 到 Hover，避免 30 次 stepSimulation 把位形拉偏
        if step == 0 and robot == "widowx" and widowx_hover_arm is not None:
            for i, j in enumerate(ARM_JOINTS):
                p.resetJointState(robot_id, j, widowx_hover_arm[i])
            # Step 0: CRITICAL ALIGNMENT 输出（HD CHECK）
            det = _widowx_detected_link_lengths_from_urdf(urdf)
            if det:
                sh = float(det.get("shoulder", 0.0)) * float(widowx_scale)
                el = float(det.get("elbow", 0.0)) * float(widowx_scale)
                tp = float(det.get("tip", 0.0)) * float(widowx_scale)
                link_s = f"shoulder={sh:.3f}, elbow={el:.3f}, tip={tp:.3f}"
            else:
                link_s = "unavailable"
            cube_z = float("nan")
            dist0 = float("nan")
            if (args.scene or "cube").lower() == "cube" and scene_info.get("objects"):
                cube_id = scene_info["objects"][0][0]
                pos, _ = p.getBasePositionAndOrientation(cube_id)
                cube_z = float(pos[2])
                dist0 = float(np.sqrt(float(pos[0]) ** 2 + float(pos[1]) ** 2))
            print(f"[HD CHECK] Link Lengths: {link_s} | Cube Z: {cube_z:.3f} | Dist: {dist0:.3f}")

        p.stepSimulation()
        if use_gui:
            time.sleep(1.0 / 60.0)

        rgb = get_rgb()
        pv = preprocess_image(rgb, processor, size=(args.cam_size, args.cam_size))
        act_norm = predict(model, pv, num_image_tokens, tokenizer, device, prompt=args.prompt)
        act_raw = denormalize(act_norm[0], mean, std)

        ls = p.getLinkState(robot_id, EE_LINK_INDEX)
        ee_pos = np.asarray(ls[0], dtype=np.float32)
        ee_quat = np.asarray(ls[1], dtype=np.float32)
        target_pos, target_quat, finger, target_pos_before_clip = action_7d_to_ik_targets(
            act_raw,
            ee_pos,
            ee_quat,
            xyz_scale=float(args.xyz_scale),
            ori_scale=float(args.ori_scale),
            ori_mode=str(args.ori_mode),
            workspace_low=WS_LOW,
            workspace_high=WS_HI,
            min_z=min_z,
            finger_max=FINGER_MAX,
        )
        if float(target_pos[2]) <= table_top_z + 0.02:
            print(f"[eval] possible table hit: target_z={target_pos[2]:.3f} table_top={table_top_z:.3f}")

        try:
            jpos = p.calculateInverseKinematics(
                robot_id,
                EE_LINK_INDEX,
                target_pos.tolist(),
                target_quat.tolist(),
                maxNumIterations=50,
                **ik_params,
            )
            if jpos is None:
                print(f"[eval] IK warning: calculateInverseKinematics returned None, target may be unreachable: pos={target_pos.tolist()}")
                jpos = [p.getJointState(robot_id, j)[0] for j in ARM_JOINTS]
                ik_success = False
            else:
                ik_success = True
        except Exception as e:
            print(f"[eval] IK warning: calculateInverseKinematics failed: {e}, target_pos={target_pos.tolist()}")
            jpos = [p.getJointState(robot_id, j)[0] for j in ARM_JOINTS]
            ik_success = False

        jpos = np.asarray(jpos, dtype=np.float32)
        # IK 可能返回全关节(含手指)，只取前 len(ARM_JOINTS) 用于手臂
        if len(jpos) > len(ARM_JOINTS):
            jpos = jpos[: len(ARM_JOINTS)]
        for i, j in enumerate(ARM_JOINTS):
            p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, targetPosition=float(jpos[i] if i < len(jpos) else jpos[-1]))
        for j in FINGER_JOINTS:
            p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, targetPosition=finger)

        vals = [p.getJointState(robot_id, j)[0] for j in ARM_JOINTS + FINGER_JOINTS]
        print(f"[step={step}] VLA Output: dx={act_raw[0]:.3f}, dy={act_raw[1]:.3f}, dz={act_raw[2]:.3f} | Gripper: g={act_raw[6]:.3f}")
        print(f"[step={step}] World Target: [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}] | IK Status: {ik_success}")
        print(f"[step={step}] Joint1 (Base Yaw, rad): {vals[0]:.4f}")

        rgb_out = np.clip(rgb, 0, 255).astype(np.uint8)
        frames.append(rgb_out)

    p.disconnect()

    out_path = args.output_mp4
    try:
        imageio.mimsave(out_path, frames, fps=args.fps, format="FFMPEG", codec="libx264")
    except Exception:
        imageio.mimsave(out_path, frames, fps=args.fps)
    print(f"[eval] Saved {out_path} ({len(frames)} frames, {args.fps} fps)")


if __name__ == "__main__":
    main()
