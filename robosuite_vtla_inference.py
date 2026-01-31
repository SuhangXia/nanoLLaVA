"""
Robosuite + Nano-VTLA 推理脚本
使用 Robosuite 仿真环境调用 VTLA API 进行视觉-语言-动作推理
"""

import argparse
import numpy as np
import requests
import base64
import io
import os
import time
from PIL import Image

parser = argparse.ArgumentParser(description="Robosuite + VTLA 推理脚本")
parser.add_argument("--api-url", type=str, default="http://localhost:8000",
                   help="VTLA API 服务地址")
parser.add_argument("--env-name", type=str, default="PickPlaceCan",
                   help="Robosuite 环境名称 (PickPlace, PickPlaceCan, Lift 等)")
parser.add_argument("--robot", type=str, default="Panda",
                   help="机器人类型")
parser.add_argument("--prompt", type=str, default="pick up the can",
                   help="语言指令")
parser.add_argument("--max-steps", type=int, default=1000,
                   help="每回合最大步数")
parser.add_argument("--num-episodes", type=int, default=1,
                   help="运行回合数")
parser.add_argument("--action-scale", type=float, default=50.0,
                   help="动作缩放因子（模型输出较小，需放大以适配 Robosuite）")
parser.add_argument("--camera-dir", type=str, default="./robosuite_camera_views",
                   help="相机图像保存目录")
parser.add_argument("--save-api-input", action="store_true", default=True,
                   help="保存发送给 API 的图像（调试用）")
parser.add_argument("--no-save-api-input", action="store_false", dest="save_api_input",
                   help="不保存 API 输入图像")
parser.add_argument("--render", action="store_true", default=True,
                   help="显示仿真窗口")
parser.add_argument("--no-render", action="store_false", dest="render",
                   help="无界面运行")
parser.add_argument("--save-every-step", type=int, default=5,
                   help="每 N 步保存一张图像到文件（0=仅保存 latest）")

args = parser.parse_args()


def main():
    # 导入 robosuite（需在 conda 环境中）
    try:
        import robosuite as suite
    except ImportError as e:
        print(f"❌ 无法导入 robosuite: {e}")
        print("请激活 robosuite 环境: conda activate robosuite_env")
        return

    # robosuite 1.5: load_composite_controller_config 在 composite_controller_factory
    load_composite_controller_config = None
    try:
        from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config
    except ImportError:
        try:
            from robosuite.controllers.composite import load_composite_controller_config
        except ImportError:
            try:
                from robosuite.controllers import load_composite_controller_config
            except ImportError:
                pass

    print("=" * 80)
    print("Robosuite + Nano-VTLA 推理")
    print("=" * 80)
    print(f"环境: {args.env_name}, 机器人: {args.robot}")
    print(f"API 地址: {args.api_url}")
    print(f"指令: {args.prompt}")
    print(f"动作缩放: {args.action_scale}x")
    print("=" * 80)

    # 检查 API
    try:
        response = requests.get(f"{args.api_url}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ API 健康检查失败: {response.status_code}")
            return
        health = response.json()
        print(f"\n✅ API 健康检查通过, 模型已加载: {health.get('model_loaded', False)}")
    except Exception as e:
        print(f"❌ 无法连接 API: {e}")
        print("请先启动 VTLA API 服务（Docker 容器内）")
        return

    # 创建保存目录
    os.makedirs(args.camera_dir, exist_ok=True)
    print(f"📷 图像保存目录: {os.path.abspath(args.camera_dir)}\n")

    # 加载控制器（robosuite 1.5: load_composite_controller_config 在 composite 子模块）
    controller_config = None
    if load_composite_controller_config is not None:
        try:
            controller_config = load_composite_controller_config(controller="BASIC")
        except Exception:
            pass

    # 创建 Robosuite 环境
    make_kwargs = dict(
        env_name=args.env_name,
        robots=args.robot,
        has_renderer=args.render,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names="robot0_eye_in_hand",  # 手眼相机（eye-in-hand）
        camera_heights=384,
        camera_widths=384,
        horizon=args.max_steps + 50,
        render_camera="robot0_eye_in_hand",
    )
    if controller_config is not None:
        make_kwargs["controller_configs"] = controller_config

    env = suite.make(**make_kwargs)

    print(f"✅ 环境创建完成: {args.env_name}")
    action_spec = env.action_spec
    if isinstance(action_spec, (tuple, list)):
        action_dim = len(action_spec[0])
    else:
        action_dim = getattr(env, "action_dim", 7)
    print(f"   动作维度: {action_dim}, 动作空间: {action_spec}")
    print("=" * 80)

    api_input_count = 0
    image_key = None

    for episode in range(args.num_episodes):
        print(f"\n{'#' * 80}")
        print(f"# Episode {episode + 1}/{args.num_episodes}")
        print(f"{'#' * 80}")

        obs = env.reset()
        done = False
        step = 0

        # 获取观测键名（robosuite 1.5: robot0_eye_in_hand_image）
        if image_key is None:
            for k in obs.keys():
                if "eye_in_hand" in k.lower() or ("robot0" in k.lower() and "image" in k.lower()):
                    image_key = k
                    break
            if image_key is None:
                image_key = "robot0_eye_in_hand_image" if "robot0_eye_in_hand_image" in obs else list(obs.keys())[0]
            print(f"使用图像观测键（手眼相机）: {image_key}")

        while not done and step < args.max_steps:
            # 1. 获取相机图像
            if image_key in obs:
                rgb = obs[image_key]
                if rgb is not None and rgb.size > 0:
                    if rgb.max() <= 1.0:
                        rgb = (rgb * 255).astype(np.uint8)
                    else:
                        rgb = rgb.astype(np.uint8)
                else:
                    rgb = np.zeros((384, 384, 3), dtype=np.uint8)
            else:
                rgb = np.zeros((384, 384, 3), dtype=np.uint8)

            # 2. 获取当前末端位姿（VTLA API 需要 [x,y,z,qx,qy,qz,qw]）
            eef_pos = obs.get("robot0_eef_pos", obs.get("robot0_eef_pos_flat", np.zeros(3)))
            if len(eef_pos) > 3:
                eef_pos = eef_pos[:3]
            eef_quat = obs.get("robot0_eef_quat", obs.get("robot0_eef_quat_flat", np.array([0, 0, 0, 1])))
            if len(eef_quat) != 4:
                eef_quat = np.array([0, 0, 0, 1])
            # Robosuite 可能用 [w,x,y,z]，API 需要 [x,y,z,qx,qy,qz,qw]
            if eef_quat[0] ** 2 > 0.5:  # w 在首位
                eef_quat = np.array([eef_quat[1], eef_quat[2], eef_quat[3], eef_quat[0]])
            current_pose = np.concatenate([eef_pos, eef_quat]).astype(np.float32)

            # 3. 保存发送给 API 的图像到本地
            if args.save_api_input:
                api_input_path = os.path.join(args.camera_dir, "api_input_latest.jpg")
                Image.fromarray(rgb).save(api_input_path, quality=95)
                if args.save_every_step > 0 and step % args.save_every_step == 0:
                    step_path = os.path.join(args.camera_dir, f"eye_in_hand_ep{episode}_step_{step:05d}.jpg")
                    Image.fromarray(rgb).save(step_path, quality=95)
                api_input_count += 1

            # 4. 调用 VTLA API
            try:
                pil_img = Image.fromarray(rgb)
                img_bytes = io.BytesIO()
                pil_img.save(img_bytes, format="JPEG")
                img_b64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")

                payload = {
                    "image": img_b64,
                    "current_pose": current_pose.tolist(),
                    "text_prompt": args.prompt,
                }

                response = requests.post(
                    f"{args.api_url}/predict_isaac",
                    json=payload,
                    timeout=30,
                )

                if response.status_code != 200:
                    print(f"  [Step {step}] ❌ API 错误: {response.status_code}")
                    action = np.zeros(action_dim)
                else:
                    result = response.json()
                    if not result.get("success", False):
                        print(f"  [Step {step}] ❌ API 返回失败: {result.get('error', '')}")
                        action = np.zeros(action_dim)
                    else:
                        delta_pose = np.array(result["delta_pose"], dtype=np.float32)
                        # delta_pose: [dx, dy, dz, drx, dry, drz]
                        # Robosuite OSC_POSE: [dx, dy, dz, droll, dpitch, dyaw, gripper]
                        scaled = delta_pose * args.action_scale
                        # gripper: -1=open, 1=close（Robosuite 惯例），默认打开
                        action = np.array([
                            scaled[0], scaled[1], scaled[2],
                            scaled[3], scaled[4], scaled[5],
                            -1.0,  # gripper open
                        ], dtype=np.float32)
                        # 裁剪/填充到实际 action 维度
                        if len(action) > action_dim:
                            action = action[:action_dim]
                        elif len(action) < action_dim:
                            action = np.pad(action, (0, action_dim - len(action)), constant_values=0)
                        # 裁剪到动作空间范围（Robosuite 通常 [-1, 1]）
                        action = np.clip(action, -1.0, 1.0)

            except Exception as e:
                print(f"  [Step {step}] ❌ API 异常: {e}")
                action = np.zeros(action_dim)

            # 5. 执行动作
            obs, reward, done, info = env.step(action)
            step += 1

            if step % 20 == 0:
                print(f"  [Step {step}] reward={reward:.3f}")

            if args.render:
                env.render()

        print(f"\n回合结束: {step} 步, reward={reward:.3f}")
        if "success" in info:
            print(f"  成功: {info['success']}")

    env.close()
    print(f"\n✅ 推理完成。相机图像已保存到: {os.path.abspath(args.camera_dir)}/")


if __name__ == "__main__":
    main()
