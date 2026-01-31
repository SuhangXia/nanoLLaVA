"""
Isaac Sim + Nano-VTLA 推理脚本（独立版本）
支持手动指定 Isaac Sim 安装路径
"""

import sys
import os
import argparse

# 解析命令行参数（需要在导入 Isaac Sim 之前）
parser = argparse.ArgumentParser(description="Isaac Sim + VTLA 推理脚本（独立版本）")
parser.add_argument("--isaac-sim-path", type=str, 
                   default="/home/suhang/datasets/isaac_sim_work",
                   help="Isaac Sim 安装路径")
parser.add_argument("--api-url", type=str, default="http://localhost:8000",
                   help="VTLA API 服务地址")
parser.add_argument("--prompt", type=str, default="pick up the red cube",
                   help="语言指令")
parser.add_argument("--max-steps", type=int, default=50,
                   help="每个回合最大步数")
parser.add_argument("--num-episodes", type=int, default=1,
                   help="运行回合数")

args = parser.parse_args()

# 添加 Isaac Sim Python 路径
isaac_sim_path = args.isaac_sim_path
if os.path.exists(isaac_sim_path):
    # 常见的 Isaac Sim Python 路径
    possible_paths = [
        os.path.join(isaac_sim_path, "python_packages"),
        os.path.join(isaac_sim_path, "kit", "python", "lib"),
        os.path.join(isaac_sim_path, "exts"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            sys.path.insert(0, path)
            print(f"✓ 添加路径: {path}")

import numpy as np
import requests
import base64
import io
from PIL import Image
import time
from scipy.spatial.transform import Rotation as R

# 尝试导入 Isaac Sim
try:
    from omni.isaac.kit import SimulationApp
    
    # 启动 Isaac Sim（无头模式或有界面）
    simulation_app = SimulationApp({
        "headless": False,  # 改为 True 可无界面运行
        "width": 1280,
        "height": 720,
    })
    
    from omni.isaac.core import World
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.objects import DynamicCuboid
    from omni.isaac.core.utils.types import ArticulationAction
    import omni.isaac.core.utils.numpy.rotations as rot_utils
    from omni.isaac.core.utils.stage import add_reference_to_stage
    import omni
    
    ISAAC_SIM_AVAILABLE = True
    print("✅ Isaac Sim 模块加载成功")
    
except ImportError as e:
    print(f"❌ Isaac Sim 未找到: {e}")
    print(f"\n请检查:")
    print(f"1. Isaac Sim 是否安装在: {isaac_sim_path}")
    print(f"2. 使用正确的 Python 启动脚本（通常是 isaac_sim_path/python.sh）")
    print(f"3. 或使用 Isaac Sim 自带的 Python 解释器")
    print(f"\n示例:")
    print(f"  {isaac_sim_path}/python.sh {__file__} --api-url http://localhost:8000")
    ISAAC_SIM_AVAILABLE = False
    sys.exit(1)


class VTLAIsaacSimClient:
    """VTLA + Isaac Sim 推理客户端（简化版）"""
    
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        self.world = None
        self.robot = None
        
    def setup_simple_scene(self):
        """初始化简化场景（不需要完整的机器人模型）"""
        print("=" * 80)
        print("初始化简化测试场景")
        print("=" * 80)
        
        # 创建 World
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        
        # 添加一个简单的立方体作为目标
        target = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/target",
                name="target",
                position=np.array([0.5, 0.0, 0.3]),
                scale=np.array([0.05, 0.05, 0.05]),
                color=np.array([1.0, 0.0, 0.0])
            )
        )
        
        # 重置
        self.world.reset()
        
        print("✅ 场景初始化完成")
        print("=" * 80)
    
    def get_test_image(self):
        """生成测试图像（红色方块）"""
        # 简化版：生成一个测试图像
        img = Image.new('RGB', (384, 384), color=(200, 50, 50))
        return np.array(img)
    
    def call_vtla_api(self, rgb_image, current_pose, text_prompt):
        """调用 VTLA API"""
        # Base64 编码
        pil_image = Image.fromarray(rgb_image)
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format='JPEG')
        img_b64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        
        # 构造请求
        payload = {
            "image": img_b64,
            "current_pose": current_pose.tolist(),
            "text_prompt": text_prompt
        }
        
        # 发送请求
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
    
    def run_simple_test(self, text_prompt="pick up the red cube", num_steps=10):
        """运行简化测试（不需要完整机器人）"""
        print("\n" + "=" * 80)
        print(f"开始简化测试: '{text_prompt}'")
        print("=" * 80)
        
        # 模拟初始位姿
        current_pose = np.array([0.3, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])  # [x,y,z,qx,qy,qz,qw]
        
        for step in range(num_steps):
            print(f"\n[Step {step}]")
            
            # 1. 获取图像（简化版：生成测试图像）
            rgb_image = self.get_test_image()
            print(f"  ✓ 图像: {rgb_image.shape}")
            
            # 2. 当前位姿
            print(f"  ✓ 当前位姿: pos={current_pose[:3]}, quat={current_pose[3:]}")
            
            # 3. 调用 API
            try:
                delta_pose = self.call_vtla_api(rgb_image, current_pose, text_prompt)
                print(f"  ✓ API 返回:")
                print(f"    - 平移: [{delta_pose[0]:.6f}, {delta_pose[1]:.6f}, {delta_pose[2]:.6f}] m")
                print(f"    - 旋转: [{delta_pose[3]:.6f}, {delta_pose[4]:.6f}, {delta_pose[5]:.6f}] rad")
                print(f"    - 平移距离: {np.linalg.norm(delta_pose[:3]):.6f} m")
            except Exception as e:
                print(f"  ❌ API 调用失败: {e}")
                break
            
            # 4. 更新位姿（简化：只累加平移）
            current_pose[:3] += delta_pose[:3]
            
            # 5. 步进仿真
            if self.world:
                self.world.step(render=True)
            
            time.sleep(0.2)
        
        print("\n" + "=" * 80)
        print(f"测试完成（共 {step+1} 步）")
        print("=" * 80)
    
    def cleanup(self):
        """清理"""
        if self.world:
            self.world.stop()
        simulation_app.close()


def main():
    print("=" * 80)
    print("Isaac Sim + Nano-VTLA 推理（独立版本）")
    print("=" * 80)
    print(f"Isaac Sim 路径: {args.isaac_sim_path}")
    print(f"API 地址: {args.api_url}")
    print(f"指令: {args.prompt}")
    print("=" * 80)
    
    # 检查 API
    try:
        response = requests.get(f"{args.api_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"\n✅ API 健康检查通过")
            print(f"   模型已加载: {health.get('model_loaded', False)}")
        else:
            print(f"\n⚠️  API 健康检查失败")
    except Exception as e:
        print(f"\n❌ 无法连接到 API: {e}")
        print("请先启动 VTLA API 服务")
        return
    
    # 创建客户端
    client = VTLAIsaacSimClient(api_url=args.api_url)
    
    try:
        # 初始化场景
        client.setup_simple_scene()
        
        # 运行测试
        for episode in range(args.num_episodes):
            print(f"\n{'#' * 80}")
            print(f"# Episode {episode + 1}/{args.num_episodes}")
            print(f"{'#' * 80}")
            
            if client.world:
                client.world.reset()
            
            client.run_simple_test(
                text_prompt=args.prompt,
                num_steps=args.max_steps
            )
            
            if episode < args.num_episodes - 1:
                time.sleep(2.0)
        
        print("\n" + "=" * 80)
        print(f"✅ 所有测试完成")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n正在清理...")
        client.cleanup()
        print("✅ 清理完成")


if __name__ == "__main__":
    main()
