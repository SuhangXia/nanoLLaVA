"""
Isaac Sim + Nano-VTLA 推理脚本
使用 Isaac Sim 仿真环境，调用 VTLA API 进行视觉-语言-动作推理
"""

import numpy as np
import requests
import base64
import io
from PIL import Image
import argparse
import time
from scipy.spatial.transform import Rotation as R

# Isaac Sim imports (需要在 Isaac Sim conda 环境中运行)
try:
    from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp({"headless": False})  # 启动 Isaac Sim
    
    from omni.isaac.core import World
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.objects import DynamicCuboid
    from omni.isaac.core.utils.types import ArticulationAction
    import omni.isaac.core.utils.numpy.rotations as rot_utils
    from omni.isaac.core.utils.stage import add_reference_to_stage
    import omni
    
    ISAAC_SIM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Isaac Sim 未安装或环境未激活: {e}")
    print("请在 Isaac Sim conda 环境中运行此脚本:")
    print("  conda activate /home/suhang/datasets/isaac_sim_work/envs/isaacsim")
    ISAAC_SIM_AVAILABLE = False


class VTLAIsaacSimClient:
    """VTLA + Isaac Sim 推理客户端"""
    
    def __init__(self, api_url="http://localhost:8000", robot_name="Franka"):
        self.api_url = api_url
        self.robot_name = robot_name
        self.world = None
        self.robot = None
        self.camera = None
        
    def setup_scene(self):
        """初始化 Isaac Sim 场景"""
        print("=" * 80)
        print("初始化 Isaac Sim 场景")
        print("=" * 80)
        
        # 创建 World
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        
        # 添加机器人（Franka Panda）
        print("[1/4] 加载 Franka Panda 机器人...")
        from omni.isaac.franka import Franka
        self.robot = self.world.scene.add(
            Franka(
                prim_path="/World/Franka",
                name="franka_robot",
                position=np.array([0.0, 0.0, 0.0])
            )
        )
        
        # 添加目标物体（红色立方体）
        print("[2/4] 添加目标物体...")
        target_cube = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/target_cube",
                name="target_cube",
                position=np.array([0.5, 0.0, 0.3]),
                scale=np.array([0.05, 0.05, 0.05]),
                color=np.array([1.0, 0.0, 0.0])  # 红色
            )
        )
        
        # 添加相机（Eye-in-Hand 或 固定相机）
        print("[3/4] 配置相机...")
        from omni.isaac.sensor import Camera
        self.camera = Camera(
            prim_path="/World/Franka/panda_hand/camera",
            position=np.array([0.0, 0.0, 0.05]),
            frequency=20,
            resolution=(384, 384)
        )
        
        # 重置场景
        print("[4/4] 重置场景...")
        self.world.reset()
        
        print("✅ 场景初始化完成")
        print("=" * 80)
        
    def get_camera_image(self):
        """
        获取相机 RGB 图像
        
        Returns:
            np.ndarray: (H, W, 3) uint8 RGB 图像
        """
        # 获取相机数据
        self.camera.initialize()
        rgb = self.camera.get_rgba()[:, :, :3]  # 去掉 alpha 通道
        
        # 转换为 uint8
        if rgb.max() <= 1.0:
            rgb = (rgb * 255).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8)
        
        return rgb
    
    def get_end_effector_pose(self):
        """
        获取末端执行器位姿
        
        Returns:
            np.ndarray: (7,) [x, y, z, qx, qy, qz, qw]
        """
        # 获取末端位置和四元数
        ee_position, ee_orientation = self.robot.end_effector.get_world_pose()
        
        # 组合为 7-DoF 位姿
        pose = np.concatenate([ee_position, ee_orientation])  # [x, y, z, qx, qy, qz, qw]
        return pose
    
    def call_vtla_api(self, rgb_image, current_pose, text_prompt):
        """
        调用 VTLA API 获取动作预测
        
        Args:
            rgb_image: np.ndarray (H, W, 3) uint8
            current_pose: np.ndarray (7,) [x, y, z, qx, qy, qz, qw]
            text_prompt: str
        
        Returns:
            delta_pose: np.ndarray (6,) [dx, dy, dz, drx, dry, drz]
        """
        # 1. 图像编码为 Base64
        pil_image = Image.fromarray(rgb_image)
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format='JPEG')
        img_b64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        
        # 2. 构造请求
        payload = {
            "image": img_b64,
            "current_pose": current_pose.tolist(),
            "text_prompt": text_prompt
        }
        
        # 3. 发送请求
        try:
            response = requests.post(
                f"{self.api_url}/predict_isaac",
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"API HTTP {response.status_code}: {response.text}")
            
            result = response.json()
            
            if not result.get("success", False):
                raise RuntimeError(f"API Error: {result.get('error', 'Unknown')}")
            
            delta_pose = np.array(result["delta_pose"], dtype=np.float32)
            return delta_pose
            
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"无法连接到 VTLA API ({self.api_url}): {e}")
    
    def apply_delta_pose(self, delta_pose, current_pose):
        """
        应用相对位姿变换到机器人
        
        Args:
            delta_pose: (6,) [dx, dy, dz, drx, dry, drz] 相对位姿
            current_pose: (7,) [x, y, z, qx, qy, qz, qw] 当前位姿
        
        Returns:
            target_pose: (7,) [x', y', z', qx', qy', qz', qw'] 目标位姿
        """
        # 提取当前位置和旋转
        current_pos = current_pose[:3]
        current_quat = current_pose[3:]  # [qx, qy, qz, qw]
        
        # 提取 delta
        delta_pos = delta_pose[:3]
        delta_rot_euler = delta_pose[3:]  # [drx, dry, drz] 欧拉角（弧度）
        
        # 1. 平移：直接相加（假设 delta 在世界坐标系）
        target_pos = current_pos + delta_pos
        
        # 2. 旋转：将 delta 欧拉角转为旋转矩阵，与当前旋转复合
        current_rot = R.from_quat([current_quat[0], current_quat[1], current_quat[2], current_quat[3]])
        delta_rot = R.from_euler('xyz', delta_rot_euler)
        target_rot = delta_rot * current_rot
        target_quat = target_rot.as_quat()  # [qx, qy, qz, qw]
        
        # 组合目标位姿
        target_pose = np.concatenate([target_pos, target_quat])
        return target_pose
    
    def execute_action(self, target_pose):
        """
        执行目标位姿（IK 或直接控制）
        
        Args:
            target_pose: (7,) [x, y, z, qx, qy, qz, qw]
        """
        target_position = target_pose[:3]
        target_orientation = target_pose[3:]
        
        # 使用 IK 控制器（如果可用）
        # 这里简化为直接设置目标位姿
        # 实际应用中应使用 IK 或 operational space controller
        
        # 示例：使用 ArticulationAction（需根据实际机器人 API 调整）
        # self.robot.end_effector.set_world_pose(target_position, target_orientation)
        
        # 或使用 IK controller
        from omni.isaac.core.controllers import BaseController
        # controller = self.robot.get_articulation_controller()
        # joint_positions = compute_ik(target_position, target_orientation)
        # controller.apply_action(ArticulationAction(joint_positions=joint_positions))
        
        # 临时：直接设置（仅用于演示）
        self.robot.end_effector.set_world_pose(target_position, target_orientation)
        
    def run_episode(self, text_prompt="pick up the red cube", max_steps=50):
        """
        运行一个推理回合
        
        Args:
            text_prompt: 语言指令
            max_steps: 最大步数
        """
        print("\n" + "=" * 80)
        print(f"开始推理回合: '{text_prompt}'")
        print("=" * 80)
        
        for step in range(max_steps):
            print(f"\n[Step {step}]")
            
            # 1. 获取当前图像
            rgb_image = self.get_camera_image()
            print(f"  ✓ 获取相机图像: {rgb_image.shape}")
            
            # 2. 获取当前末端位姿
            current_pose = self.get_end_effector_pose()
            print(f"  ✓ 当前位姿: pos={current_pose[:3]}, quat={current_pose[3:]}")
            
            # 3. 调用 VTLA API
            try:
                delta_pose = self.call_vtla_api(rgb_image, current_pose, text_prompt)
                print(f"  ✓ API 返回 delta_pose: {delta_pose}")
                print(f"    - 平移: [{delta_pose[0]:.6f}, {delta_pose[1]:.6f}, {delta_pose[2]:.6f}] m")
                print(f"    - 旋转: [{delta_pose[3]:.6f}, {delta_pose[4]:.6f}, {delta_pose[5]:.6f}] rad")
            except Exception as e:
                print(f"  ❌ API 调用失败: {e}")
                break
            
            # 4. 计算目标位姿
            target_pose = self.apply_delta_pose(delta_pose, current_pose)
            print(f"  ✓ 目标位姿: pos={target_pose[:3]}, quat={target_pose[3:]}")
            
            # 5. 执行动作
            try:
                self.execute_action(target_pose)
                print(f"  ✓ 动作已执行")
            except Exception as e:
                print(f"  ❌ 动作执行失败: {e}")
                break
            
            # 6. 步进仿真
            self.world.step(render=True)
            time.sleep(0.1)  # 10Hz 控制频率
            
            # 7. 检查是否完成（这里简化，实际应检查任务成功条件）
            # 例如：物体是否被抓取、是否到达目标位置等
            
        print("\n" + "=" * 80)
        print(f"回合结束（共 {step+1} 步）")
        print("=" * 80)
    
    def cleanup(self):
        """清理资源"""
        if self.world:
            self.world.stop()
        simulation_app.close()


def main():
    parser = argparse.ArgumentParser(description="Isaac Sim + VTLA 推理脚本")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000",
                       help="VTLA API 服务地址")
    parser.add_argument("--prompt", type=str, default="pick up the red cube",
                       help="语言指令")
    parser.add_argument("--max-steps", type=int, default=50,
                       help="每个回合最大步数")
    parser.add_argument("--num-episodes", type=int, default=1,
                       help="运行回合数")
    
    args = parser.parse_args()
    
    if not ISAAC_SIM_AVAILABLE:
        print("\n❌ Isaac Sim 不可用，退出")
        return
    
    print("=" * 80)
    print("Isaac Sim + Nano-VTLA 推理")
    print("=" * 80)
    print(f"API 地址: {args.api_url}")
    print(f"指令: {args.prompt}")
    print(f"最大步数: {args.max_steps}")
    print(f"回合数: {args.num_episodes}")
    print("=" * 80)
    
    # 检查 API 健康状态
    try:
        response = requests.get(f"{args.api_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"\n✅ API 健康检查通过")
            print(f"   状态: {health.get('status', 'unknown')}")
            print(f"   模型已加载: {health.get('model_loaded', False)}")
        else:
            print(f"\n⚠️  API 健康检查失败: {response.status_code}")
    except Exception as e:
        print(f"\n❌ 无法连接到 API: {e}")
        print(f"请确保 VTLA API 服务正在运行:")
        print(f"  docker exec -it nanollava_vtla_new bash")
        print(f"  cd /workspace/nanoLLaVA")
        print(f"  python serve_vtla_api.py --checkpoint ... --port 8000")
        return
    
    # 创建客户端
    client = VTLAIsaacSimClient(api_url=args.api_url)
    
    try:
        # 初始化场景
        client.setup_scene()
        
        # 运行推理回合
        for episode in range(args.num_episodes):
            print(f"\n{'#' * 80}")
            print(f"# Episode {episode + 1}/{args.num_episodes}")
            print(f"{'#' * 80}")
            
            # 重置场景
            client.world.reset()
            
            # 运行推理
            client.run_episode(
                text_prompt=args.prompt,
                max_steps=args.max_steps
            )
            
            # 等待一下再开始下一回合
            if episode < args.num_episodes - 1:
                time.sleep(2.0)
        
        print("\n" + "=" * 80)
        print(f"✅ 所有回合完成 ({args.num_episodes} 个)")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理
        print("\n正在清理资源...")
        client.cleanup()
        print("✅ 清理完成")


if __name__ == "__main__":
    main()
