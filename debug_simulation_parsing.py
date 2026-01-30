"""
调试脚本：模拟仿真环境的 API 调用和解析
帮助找出为什么仿真环境收到 [0 0 0] 而 API 实际返回非零值
"""

import requests
import numpy as np
from PIL import Image
import io
import json


def simulate_simulation_parsing(api_url="http://localhost:8000"):
    """
    模拟仿真环境的解析方式，找出问题
    """
    print("=" * 80)
    print("模拟仿真环境 API 调用和解析")
    print("=" * 80)
    
    # 1. 创建测试图像
    test_image = Image.new('RGB', (384, 384), color='red')
    img_bytes = io.BytesIO()
    test_image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # 2. 发送请求（模拟仿真环境）
    files = {'image': ('image.jpg', img_bytes, 'image/jpeg')}
    data = {
        'instruction': 'pick up the red cup',
        'use_dummy_tactile': True
    }
    
    print(f"\n[1] 发送请求到: {api_url}/predict")
    print(f"    Instruction: {data['instruction']}")
    
    try:
        response = requests.post(f"{api_url}/predict", files=files, data=data, timeout=30)
        
        print(f"\n[2] HTTP 状态码: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ HTTP 错误: {response.text}")
            return
        
        # 3. 解析 JSON（模拟仿真环境可能的各种解析方式）
        result = response.json()
        
        print(f"\n[3] 原始 JSON 响应:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        print(f"\n[4] 测试不同的解析方式:")
        print("-" * 80)
        
        # 方式1: 直接取 action（正确方式）
        print("\n方式1: result['action'] (推荐)")
        try:
            action1 = result['action']
            print(f"  ✓ 成功: {action1}")
            print(f"  类型: {type(action1)}")
            if isinstance(action1, list):
                action1_arr = np.array(action1)
                print(f"  转换为 numpy: {action1_arr}")
                print(f"  是否全零: {np.allclose(action1_arr, 0)}")
                if not np.allclose(action1_arr, 0):
                    print(f"  ✅ 非零值！")
                else:
                    print(f"  ❌ 全零！")
        except KeyError as e:
            print(f"  ❌ 错误: 缺少键 {e}")
        
        # 方式2: 从 action_breakdown 提取
        print("\n方式2: 从 action_breakdown 提取")
        try:
            breakdown = result['action_breakdown']
            action2 = np.array([
                breakdown['translation']['dx'],
                breakdown['translation']['dy'],
                breakdown['translation']['dz'],
                breakdown['rotation']['droll'],
                breakdown['rotation']['dpitch'],
                breakdown['rotation']['dyaw'],
                breakdown['gripper']['value']
            ])
            print(f"  ✓ 成功: {action2}")
            print(f"  是否全零: {np.allclose(action2, 0)}")
        except KeyError as e:
            print(f"  ❌ 错误: 缺少键 {e}")
        
        # 方式3: 检查是否有嵌套结构
        print("\n方式3: 检查嵌套结构")
        if 'data' in result:
            print(f"  ✓ 找到 result['data']: {result['data']}")
            if isinstance(result['data'], dict) and 'action' in result['data']:
                action3 = np.array(result['data']['action'])
                print(f"  ✓ result['data']['action']: {action3}")
        else:
            print(f"  - 没有 result['data'] 字段")
        
        # 方式4: 检查 success 字段
        print("\n方式4: 检查 success 字段")
        if 'success' in result:
            print(f"  ✓ result['success']: {result['success']}")
            if not result['success']:
                print(f"  ⚠️  Warning: success=False, action 可能无效")
                print(f"     错误信息: {result.get('error', 'N/A')}")
        else:
            print(f"  - 没有 success 字段")
        
        # 方式5: 检查常见的错误解析方式
        print("\n方式5: 常见错误解析方式检查")
        
        # 错误1: 期望 result['data']['action']
        if 'data' not in result:
            print(f"  ⚠️  如果代码使用 result['data']['action']，会报 KeyError")
        
        # 错误2: 期望 result['prediction']
        if 'prediction' not in result:
            print(f"  ⚠️  如果代码使用 result['prediction']，会报 KeyError")
        
        # 错误3: 没有转换为 numpy
        if 'action' in result and isinstance(result['action'], list):
            print(f"  ⚠️  如果代码直接使用 list 而不是 numpy array，可能有问题")
        
        # 6. 生成仿真环境应该使用的代码
        print("\n" + "=" * 80)
        print("仿真环境应该使用的解析代码:")
        print("=" * 80)
        print("""
# ✅ 正确的解析方式
response = requests.post(api_url + '/predict', files=files, data=data)
result = response.json()

if result.get('success', False):
    # 方式1: 直接使用 action 数组（推荐）
    action = np.array(result['action'], dtype=np.float32)
    
    # 验证
    if action.shape != (7,):
        raise ValueError(f"Expected action shape (7,), got {action.shape}")
    
    # 提取分量
    dx, dy, dz = action[0], action[1], action[2]
    droll, dpitch, dyaw = action[3], action[4], action[5]
    gripper = action[6]
    
    print(f"Action: [{dx:.6f}, {dy:.6f}, {dz:.6f}, {droll:.6f}, {dpitch:.6f}, {dyaw:.6f}, {gripper:.4f}]")
else:
    raise ValueError(f"API error: {result.get('error', 'Unknown')}")
        """)
        
        # 7. 检查是否有问题
        print("\n" + "=" * 80)
        print("诊断结果:")
        print("=" * 80)
        
        if 'action' in result:
            action = np.array(result['action'])
            if np.allclose(action, 0):
                print("❌ 问题: API 返回的 action 是全零")
                print("   可能原因:")
                print("   1. 模型输出问题")
                print("   2. 归一化/反归一化问题")
                print("   3. 模型未正确加载")
            else:
                print("✅ API 返回非零动作值")
                print(f"   如果仿真环境收到 [0 0 0]，问题在解析代码")
                print(f"   请检查仿真环境的解析逻辑")
        else:
            print("❌ 问题: API 响应中没有 'action' 字段")
            print(f"   可用字段: {list(result.keys())}")
        
    except requests.exceptions.ConnectionError:
        print(f"\n❌ 无法连接到 API: {api_url}")
        print("   请确保 API 服务正在运行:")
        print("   python serve_vtla_api.py --checkpoint ... --port 8000")
    except Exception as e:
        import traceback
        print(f"\n❌ 错误: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", type=str, default="http://localhost:8000",
                       help="API 服务地址")
    
    args = parser.parse_args()
    
    simulate_simulation_parsing(api_url=args.api_url)
