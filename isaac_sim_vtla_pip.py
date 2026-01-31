"""
Isaac Sim + Nano-VTLA 推理脚本
适配通过 pip 安装的 Isaac Sim 4.x+
修复：使用 IK 控制器，修正相机朝向
"""

import argparse
import numpy as np
import time

# 解析参数（需要在导入 isaacsim 之前）
parser = argparse.ArgumentParser(description="Isaac Sim + VTLA 推理脚本 (pip 版本)")
parser.add_argument("--api-url", type=str, default="http://localhost:8000",
                   help="VTLA API 服务地址")
parser.add_argument("--prompt", type=str, default="pick up the red cube",
                   help="语言指令")
parser.add_argument("--max-steps", type=int, default=2000,
                   help="每个回合最大步数")
parser.add_argument("--num-episodes", type=int, default=1,
                   help="运行回合数")
parser.add_argument("--headless", action="store_true",
                   help="无界面模式运行")
parser.add_argument("--action-scale", type=float, default=10.0,
                   help="动作缩放因子（模型输出的 delta 很小，需要放大）")
parser.add_argument("--save-camera", action="store_true", default=True,
                   help="保存相机视角图像到文件")
parser.add_argument("--no-save-camera", action="store_false", dest="save_camera",
                   help="不保存相机视角图像")
parser.add_argument("--camera-dir", type=str, default="./camera_views",
                   help="相机图像保存目录")
parser.add_argument("--save-api-input", action="store_true", default=True,
                   help="保存发送给推理 API 的图像（用于调试）")
parser.add_argument("--no-save-api-input", action="store_false", dest="save_api_input",
                   help="不保存 API 输入图像")

args = parser.parse_args()

# ============================================================
# 启动 Isaac Sim（pip 安装版本）
# ============================================================
print("=" * 80)
print("启动 Isaac Sim（pip 安装版本）")
print("=" * 80)

try:
    from isaacsim import SimulationApp
    
    config = {
        "headless": args.headless,
        "width": 1280,
        "height": 720,
        "anti_aliasing": 0,
    }
    
    print(f"配置: {config}")
    simulation_app = SimulationApp(config)
    print("✅ SimulationApp 启动成功")
    
except ImportError as e:
    print(f"❌ 无法导入 isaacsim: {e}")
    exit(1)

# ============================================================
# 导入其他依赖（必须在 SimulationApp 启动之后）
# ============================================================
import requests
import base64
import io
from PIL import Image
from scipy.spatial.transform import Rotation as R
import os

# 导入 Isaac Sim 核心模块
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, VisualCuboid
from omni.isaac.core.prims import XFormPrim
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.types import ArticulationAction
print("✅ Isaac Sim 核心模块加载成功")

# 导入 Franka 机器人和控制器
try:
    from omni.isaac.franka import Franka
    from omni.isaac.franka.controllers import RMPFlowController
    FRANKA_AVAILABLE = True
    print("✅ Franka 机器人和 RMPFlow 控制器可用")
except ImportError as e:
    print(f"⚠️  Franka 控制器不可用: {e}")
    FRANKA_AVAILABLE = False

# 尝试导入 IK 控制器（备用方案）
try:
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver
    IK_AVAILABLE = True
    print("✅ IK 求解器可用")
except ImportError:
    IK_AVAILABLE = False
    print("⚠️  IK 求解器不可用")


class VTLAIsaacClient:
    """VTLA + Isaac Sim 推理客户端（使用 RMPFlow 控制器）"""
    
    def __init__(self, api_url="http://localhost:8000", action_scale=10.0, save_camera=True, camera_dir="./camera_views", save_api_input=True):
        self.api_url = api_url
        self.action_scale = action_scale
        self.save_camera = save_camera
        self.camera_dir = camera_dir
        self.save_api_input = save_api_input
        self.frame_count = 0
        self.api_input_count = 0
        
        # 创建保存目录
        os.makedirs(self.camera_dir, exist_ok=True)
        print(f"📷 相机/API 输入图像保存目录: {os.path.abspath(self.camera_dir)}")
        self.world = None
        self.robot = None
        self.controller = None
        self.camera = None
        self.target = None
        
        # 目标末端位姿
        self.target_position = None
        self.target_orientation = None
        
    def setup_scene(self):
        """初始化场景"""
        print("\n" + "=" * 80)
        print("初始化场景")
        print("=" * 80)
        
        # 创建 World
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        print("[1/5] ✓ World 创建完成")
        
        # 添加目标物体（红色立方体）
        self.target = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/target_cube",
                name="target_cube",
                position=np.array([0.5, 0.0, 0.05]),
                scale=np.array([0.05, 0.05, 0.05]),
                color=np.array([1.0, 0.0, 0.0])  # 红色
            )
        )
        print("[2/5] ✓ 目标物体创建完成")
        
        # 添加 Franka 机器人
        if FRANKA_AVAILABLE:
            self.robot = self.world.scene.add(
                Franka(
                    prim_path="/World/Franka",
                    name="franka_robot",
                    position=np.array([0.0, 0.0, 0.0])
                )
            )
            print("[3/5] ✓ Franka 机器人创建完成")
        else:
            print("[3/5] ❌ Franka 机器人不可用")
            return
        
        # 添加相机（从斜上方俯视工作区）
        # 位置：在机器人侧前方上方
        camera_position = np.array([0.7, -0.5, 0.8])  # x=前, y=侧, z=高
        target = np.array([0.4, 0.0, 0.15])  # 工作区中心（机器人+目标区域）
        
        # 使用 Isaac Sim lookAt 或 rot_utils 设置朝向（与官方示例一致）
        camera_quat_wxyz = None
        try:
            from pxr import Gf
            from isaacsim.core.includes.math import lookAt
            camera_pos_gf = Gf.Vec3f(float(camera_position[0]), float(camera_position[1]), float(camera_position[2]))
            target_gf = Gf.Vec3f(float(target[0]), float(target[1]), float(target[2]))
            up_gf = Gf.Vec3f(0, 0, 1)  # Isaac Sim Z-up
            quat_gf = lookAt(camera_pos_gf, target_gf, up_gf)
            camera_quat_wxyz = np.array([quat_gf.GetReal(), quat_gf.GetImaginary()[0], quat_gf.GetImaginary()[1], quat_gf.GetImaginary()[2]])
            print("     使用 lookAt 设置相机朝向")
        except Exception as e:
            print(f"     lookAt 不可用 ({e})，改用 rot_utils")
            try:
                import isaacsim.core.utils.numpy.rotations as rot_utils
                camera_quat_wxyz = rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True)
                print("     使用 rot_utils.euler_angles_to_quats([0,90,0])")
            except Exception as e2:
                print(f"     rot_utils 不可用 ({e2})，改用 omni.isaac.core")
                try:
                    import omni.isaac.core.utils.numpy.rotations as rot_utils
                    camera_quat_wxyz = rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True)
                    print("     使用 omni.isaac rot_utils.euler_angles_to_quats([0,90,0])")
                except Exception as e3:
                    print(f"     回退到 scipy 欧拉角 ({e3})")
                    camera_rot = R.from_euler('xyz', [0, 90, 0], degrees=True)
                    camera_quat = camera_rot.as_quat()  # xyzw
                    camera_quat_wxyz = np.array([camera_quat[3], camera_quat[0], camera_quat[1], camera_quat[2]])
        
        self.camera = Camera(
            prim_path="/World/Camera",
            position=camera_position,
            frequency=30,
            resolution=(384, 384),
            orientation=camera_quat_wxyz
        )
        
        # 设置焦距为 15mm（广角）- 需要在创建后通过 prim 属性设置
        try:
            from pxr import UsdGeom
            camera_prim = self.world.stage.GetPrimAtPath("/World/Camera")
            if camera_prim.IsValid():
                camera_geom = UsdGeom.Camera(camera_prim)
                camera_geom.GetFocalLengthAttr().Set(15.0)
                print("     焦距设置为 15mm")
        except Exception as e:
            print(f"     ⚠️  焦距设置失败: {e}")
        print(f"[4/5] ✓ 相机创建完成 (位置: {camera_position})")
        
        # 重置场景
        self.world.reset()
        
        # 初始化相机
        self.camera.initialize()
        
        # 创建 RMPFlow 控制器（用于 IK 和运动规划）
        if FRANKA_AVAILABLE:
            try:
                self.controller = RMPFlowController(
                    name="rmpflow_controller",
                    robot_articulation=self.robot
                )
                print("[5/5] ✓ RMPFlow 控制器创建完成")
            except Exception as e:
                print(f"[5/5] ⚠️  RMPFlow 控制器创建失败: {e}")
                self.controller = None
        
        # 获取初始末端位姿
        self.target_position, self.target_orientation = self.robot.end_effector.get_world_pose()
        print(f"     初始末端位置: {self.target_position}")
        
        # 预热 60+ 帧（确保 render product 数据就绪，参考 Isaac Lab #1088）
        for _ in range(60):
            self.world.step(render=True)
        
        print("\n✅ 场景初始化完成")
        print("=" * 80)
    
    def get_camera_image(self):
        """获取相机图像"""
        rgb = None
        if self.camera:
            try:
                # 1. 先渲染一帧以更新相机
                self.world.step(render=True)
                
                # 2. 若 Camera 有 update(dt)，显式更新传感器数据
                if hasattr(self.camera, 'update'):
                    dt = 1.0 / 30.0  # 假设 30Hz
                    self.camera.update(dt=dt)
                
                # 3. 获取 RGBA 图像
                rgba = self.camera.get_rgba()
                if rgba is not None and rgba.size > 0:
                    rgb = rgba[:, :, :3]
                    if rgb.max() <= 1.0:
                        rgb = (rgb * 255).astype(np.uint8)
                    else:
                        rgb = rgb.astype(np.uint8)
            except Exception as e:
                print(f"⚠️  获取相机图像失败: {e}")
        
        # 回退：生成测试图像
        if rgb is None:
            rgb = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
        
        # 保存相机图像到文件（如果启用）
        if self.save_camera and rgb is not None:
            # 每 5 帧保存一次
            if self.frame_count % 5 == 0:
                pil_img = Image.fromarray(rgb)
                save_path = os.path.join(self.camera_dir, f"frame_{self.frame_count:05d}.jpg")
                pil_img.save(save_path, quality=90)
                # 同时保存最新帧（方便实时查看）
                latest_path = os.path.join(self.camera_dir, "latest.jpg")
                pil_img.save(latest_path, quality=90)
            self.frame_count += 1
        
        return rgb
    
    def get_current_pose(self):
        """获取当前末端位姿"""
        if self.robot:
            try:
                ee_pos, ee_quat = self.robot.end_effector.get_world_pose()
                # Isaac Sim 返回 wxyz，转换为 xyzw
                ee_quat_xyzw = np.array([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]])
                return np.concatenate([ee_pos, ee_quat_xyzw])
            except Exception as e:
                print(f"⚠️  获取位姿失败: {e}")
        
        return np.array([0.3, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])
    
    def call_vtla_api(self, rgb_image, current_pose, text_prompt):
        """调用 VTLA API"""
        # Base64 编码
        pil_image = Image.fromarray(rgb_image)
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format='JPEG')
        img_b64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        
        payload = {
            "image": img_b64,
            "current_pose": current_pose.tolist(),
            "text_prompt": text_prompt
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/predict_isaac",
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
            
            result = response.json()
            
            if not result.get("success", False):
                raise RuntimeError(f"API Error: {result.get('error', 'Unknown')}")
            
            return np.array(result["delta_pose"], dtype=np.float32)
            
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"无法连接到 API ({self.api_url}): {e}")
    
    def apply_delta_and_move(self, delta_pose):
        """
        应用相对位姿并使用 RMPFlow 控制器移动机器人
        
        Args:
            delta_pose: (6,) [dx, dy, dz, drx, dry, drz]
        """
        if self.robot is None or self.controller is None:
            print("⚠️  机器人或控制器不可用")
            return
        
        # 放大 delta（模型输出的值很小，约 0.0007m）
        scaled_delta_pos = delta_pose[:3] * self.action_scale
        scaled_delta_rot = delta_pose[3:] * self.action_scale
        
        # 更新目标位置
        self.target_position = self.target_position + scaled_delta_pos
        
        # 更新目标朝向（简化：只累加欧拉角）
        current_rot = R.from_quat([
            self.target_orientation[1],  # x
            self.target_orientation[2],  # y
            self.target_orientation[3],  # z
            self.target_orientation[0]   # w
        ])
        delta_rot = R.from_euler('xyz', scaled_delta_rot)
        new_rot = delta_rot * current_rot
        new_quat = new_rot.as_quat()  # xyzw
        # 转回 wxyz
        self.target_orientation = np.array([new_quat[3], new_quat[0], new_quat[1], new_quat[2]])
        
        # 使用 RMPFlow 控制器计算关节动作
        actions = self.controller.forward(
            target_end_effector_position=self.target_position,
            target_end_effector_orientation=self.target_orientation
        )
        
        # 应用关节动作
        self.robot.apply_action(actions)
        
        return scaled_delta_pos
    
    def run_episode(self, text_prompt, max_steps):
        """运行一个推理回合"""
        print("\n" + "=" * 80)
        print(f"开始推理: '{text_prompt}'")
        print(f"动作缩放因子: {self.action_scale}")
        print("=" * 80)
        
        # 重置目标位姿为当前位姿
        self.target_position, self.target_orientation = self.robot.end_effector.get_world_pose()
        
        for step in range(max_steps):
            print(f"\n[Step {step}]")
            
            # 1. 获取图像
            rgb_image = self.get_camera_image()
            print(f"  ✓ 图像: {rgb_image.shape}")
            
            # 2. 获取当前位姿
            current_pose = self.get_current_pose()
            print(f"  ✓ 当前位姿: pos={current_pose[:3].round(4)}")
            
            # 3. 保存发送给 API 的图像（调试用：查看模型实际看到的内容）
            if self.save_api_input:
                os.makedirs(self.camera_dir, exist_ok=True)
                api_input_path = os.path.join(self.camera_dir, "api_input_latest.jpg")
                Image.fromarray(rgb_image).save(api_input_path, quality=95)
                if self.api_input_count % 10 == 0:
                    step_path = os.path.join(self.camera_dir, f"api_input_step_{self.api_input_count:05d}.jpg")
                    Image.fromarray(rgb_image).save(step_path, quality=95)
                self.api_input_count += 1
                print(f"  ✓ 已保存 API 输入图像: {api_input_path}")
            
            # 4. 调用 API
            try:
                delta_pose = self.call_vtla_api(rgb_image, current_pose, text_prompt)
                print(f"  ✓ 模型输出 delta_pose:")
                print(f"    - 平移: [{delta_pose[0]:.6f}, {delta_pose[1]:.6f}, {delta_pose[2]:.6f}] m")
                print(f"    - 旋转: [{delta_pose[3]:.6f}, {delta_pose[4]:.6f}, {delta_pose[5]:.6f}] rad")
            except Exception as e:
                print(f"  ❌ API 失败: {e}")
                break
            
            # 5. 应用 delta 并移动机器人（使用 RMPFlow）
            scaled_delta = self.apply_delta_and_move(delta_pose)
            print(f"  ✓ 缩放后 delta: [{scaled_delta[0]:.4f}, {scaled_delta[1]:.4f}, {scaled_delta[2]:.4f}] m")
            print(f"  ✓ 目标位置: {self.target_position.round(4)}")
            
            # 6. 步进仿真（多步以让控制器跟踪）
            for _ in range(5):
                self.world.step(render=True)
            
            # 7. 检查实际位置
            actual_pos, _ = self.robot.end_effector.get_world_pose()
            error = np.linalg.norm(actual_pos - self.target_position)
            print(f"  ✓ 实际位置: {actual_pos.round(4)} (误差: {error:.4f}m)")
            
            time.sleep(0.05)
        
        print("\n" + "=" * 80)
        print(f"回合结束（共 {step+1} 步）")
        print("=" * 80)
    
    def cleanup(self):
        """清理"""
        if self.controller:
            self.controller.reset()
        if self.world:
            self.world.stop()
        simulation_app.close()


def check_api_health(api_url):
    """检查 API 健康状态"""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"\n✅ API 健康检查通过")
            print(f"   模型已加载: {health.get('model_loaded', False)}")
            print(f"   GPU 内存: {health.get('gpu_memory', 'N/A')}")
            return True
        else:
            print(f"\n⚠️  API 响应异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"\n❌ 无法连接到 API: {e}")
        return False


def main():
    print("\n" + "=" * 80)
    print("Isaac Sim + Nano-VTLA 推理（使用 RMPFlow 控制器）")
    print("=" * 80)
    print(f"API 地址: {args.api_url}")
    print(f"指令: {args.prompt}")
    print(f"最大步数: {args.max_steps}")
    print(f"动作缩放: {args.action_scale}x")
    print("=" * 80)
    
    # 检查 API
    if not check_api_health(args.api_url):
        simulation_app.close()
        return
    
    # 创建客户端
    client = VTLAIsaacClient(
        api_url=args.api_url, 
        action_scale=args.action_scale,
        save_camera=args.save_camera,
        camera_dir=args.camera_dir,
        save_api_input=args.save_api_input
    )
    
    try:
        # 初始化场景
        client.setup_scene()
        
        # 运行推理回合
        for episode in range(args.num_episodes):
            print(f"\n{'#' * 80}")
            print(f"# Episode {episode + 1}/{args.num_episodes}")
            print(f"{'#' * 80}")
            
            client.world.reset()
            
            # 重新初始化控制器
            if client.controller:
                client.controller.reset()
            
            client.run_episode(
                text_prompt=args.prompt,
                max_steps=args.max_steps
            )
            
            if episode < args.num_episodes - 1:
                time.sleep(2.0)
        
        print("\n" + "=" * 80)
        print(f"✅ 所有回合完成")
        print("=" * 80)
        
        # 保持窗口打开
        print("\n按 Ctrl+C 退出...")
        while True:
            client.world.step(render=True)
            time.sleep(0.1)
        
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
