"""
Nano-VTLA API 客户端示例
展示如何正确调用 API 并解析响应数据
"""

import requests
import numpy as np
from PIL import Image
import io
import json


def call_vtla_api(image, instruction, api_url="http://localhost:8000", use_dummy_tactile=True):
    """
    调用 VTLA API 获取动作预测
    
    Args:
        image: PIL Image 或 numpy array (H, W, 3)
        instruction: str, 语言指令
        api_url: str, API 基础 URL（不含 /predict）
        use_dummy_tactile: bool, 是否使用 dummy 触觉
    
    Returns:
        dict: {
            'success': bool,
            'action': np.array, shape (7,), [dx, dy, dz, droll, dpitch, dyaw, gripper]
            'raw_response': dict, 原始 JSON 响应
        }
    """
    # 转换图像
    if isinstance(image, np.ndarray):
        if image.max() <= 1.0:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        image = Image.fromarray(image)
    
    # 准备图像字节流
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # 准备请求
    files = {'image': ('image.jpg', img_bytes, 'image/jpeg')}
    data = {
        'instruction': instruction if instruction else "What action should the robot take?",
        'use_dummy_tactile': use_dummy_tactile
    }
    
    # 发送请求
    try:
        response = requests.post(
            f"{api_url}/predict",
            files=files,
            data=data,
            timeout=30
        )
        
        # 检查 HTTP 状态码
        if response.status_code != 200:
            return {
                'success': False,
                'error': f"HTTP {response.status_code}: {response.text}",
                'action': None,
                'raw_response': None
            }
        
        # 解析 JSON
        result = response.json()
        
        # 检查 API 返回的 success 字段
        if not result.get('success', False):
            return {
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'action': None,
                'raw_response': result
            }
        
        # 解析动作数组
        action_list = result.get('action', None)
        if action_list is None:
            return {
                'success': False,
                'error': "Response missing 'action' field",
                'action': None,
                'raw_response': result
            }
        
        # 转换为 numpy array
        action = np.array(action_list, dtype=np.float32)
        
        # 验证形状
        if action.shape != (7,):
            return {
                'success': False,
                'error': f"Expected action shape (7,), got {action.shape}",
                'action': None,
                'raw_response': result
            }
        
        # 检查是否全零（可能是模型问题）
        if np.allclose(action, 0.0, atol=1e-6):
            print("⚠️  Warning: Action is all zeros! This might indicate:")
            print("   1. Model not properly loaded")
            print("   2. Model output issue")
            print("   3. Normalization problem")
        
        return {
            'success': True,
            'action': action,
            'raw_response': result,
            'error': None
        }
        
    except requests.exceptions.Timeout:
        return {
            'success': False,
            'error': "Request timeout (30s)",
            'action': None,
            'raw_response': None
        }
    except requests.exceptions.ConnectionError:
        return {
            'success': False,
            'error': f"Connection error: Cannot connect to {api_url}",
            'action': None,
            'raw_response': None
        }
    except json.JSONDecodeError as e:
        return {
            'success': False,
            'error': f"JSON decode error: {e}",
            'action': None,
            'raw_response': None
        }
    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': f"Unexpected error: {e}\n{traceback.format_exc()}",
            'action': None,
            'raw_response': None
        }


def print_action_details(result):
    """打印动作详情"""
    if not result['success']:
        print(f"❌ Error: {result['error']}")
        return
    
    action = result['action']
    
    print("=" * 80)
    print("动作预测结果")
    print("=" * 80)
    print(f"平移 (Translation):")
    print(f"  dx: {action[0]:.6f} m")
    print(f"  dy: {action[1]:.6f} m")
    print(f"  dz: {action[2]:.6f} m")
    print(f"  总距离: {np.linalg.norm(action[:3]):.6f} m")
    print()
    print(f"旋转 (Rotation):")
    print(f"  droll:  {action[3]:.6f} rad ({np.degrees(action[3]):.2f}°)")
    print(f"  dpitch: {action[4]:.6f} rad ({np.degrees(action[4]):.2f}°)")
    print(f"  dyaw:   {action[5]:.6f} rad ({np.degrees(action[5]):.2f}°)")
    print(f"  总旋转: {np.linalg.norm(action[3:6]):.6f} rad ({np.degrees(np.linalg.norm(action[3:6])):.2f}°)")
    print()
    print(f"夹爪 (Gripper): {action[6]:.4f} ({'闭合' if action[6] > 0.5 else '打开'})")
    print()
    print(f"完整动作向量:")
    print(f"  [{action[0]:.6f}, {action[1]:.6f}, {action[2]:.6f}, "
          f"{action[3]:.6f}, {action[4]:.6f}, {action[5]:.6f}, {action[6]:.4f}]")
    print("=" * 80)


# 仿真环境集成示例
def simulation_loop_example(api_url="http://localhost:8000"):
    """
    仿真环境循环示例
    展示如何在仿真循环中调用 API
    """
    print("=" * 80)
    print("仿真环境集成示例")
    print("=" * 80)
    
    # 模拟仿真环境
    for step in range(5):
        print(f"\n[Step {step}]")
        
        # 1. 获取当前图像（这里用测试图像代替）
        current_image = Image.new('RGB', (384, 384), color='red')
        
        # 2. 调用 API
        result = call_vtla_api(
            image=current_image,
            instruction="pick up the red cup",
            api_url=api_url,
            use_dummy_tactile=True
        )
        
        if result['success']:
            action = result['action']
            
            # 3. 提取动作分量
            dx, dy, dz = action[0], action[1], action[2]
            droll, dpitch, dyaw = action[3], action[4], action[5]
            gripper = action[6]
            
            # 4. 打印（仿真环境可以在这里执行动作）
            print(f"  ✓ API 调用成功")
            print(f"  - Camera frame Δpos: [{dx:.6f}, {dy:.6f}, {dz:.6f}]")
            print(f"  - Camera frame Δrot: [{droll:.6f}, {dpitch:.6f}, {dyaw:.6f}]")
            print(f"  - Gripper: {'CLOSED' if gripper > 0.5 else 'OPEN'} ({gripper:.4f})")
            
            # 5. 在这里执行动作（转换为基坐标系等）
            # base_frame_action = transform_to_base_frame(action, camera_pose)
            # execute_action(base_frame_action)
            
        else:
            print(f"  ❌ API 调用失败: {result['error']}")
            break


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Nano-VTLA API 客户端示例")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000",
                       help="API 服务地址")
    parser.add_argument("--test", action="store_true",
                       help="运行测试")
    parser.add_argument("--sim-loop", action="store_true",
                       help="运行仿真循环示例")
    
    args = parser.parse_args()
    
    if args.test:
        # 简单测试
        print("=" * 80)
        print("API 客户端测试")
        print("=" * 80)
        
        test_image = Image.new('RGB', (384, 384), color='red')
        result = call_vtla_api(
            image=test_image,
            instruction="pick up the red cup",
            api_url=args.api_url
        )
        
        print_action_details(result)
        
        if result['success']:
            print("\n✅ 测试成功！")
        else:
            print(f"\n❌ 测试失败: {result['error']}")
    
    elif args.sim_loop:
        simulation_loop_example(api_url=args.api_url)
    
    else:
        # 默认：运行测试
        test_image = Image.new('RGB', (384, 384), color='red')
        result = call_vtla_api(
            image=test_image,
            instruction="pick up the red cup",
            api_url=args.api_url
        )
        
        print_action_details(result)
