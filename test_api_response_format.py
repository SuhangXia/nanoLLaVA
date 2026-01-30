"""
测试 API 响应格式，帮助调试仿真环境的数据解析问题
"""

import requests
import json
import numpy as np
from PIL import Image
import io

# 创建一个测试图像（384x384 RGB）
test_image = Image.new('RGB', (384, 384), color='red')
img_bytes = io.BytesIO()
test_image.save(img_bytes, format='JPEG')
img_bytes.seek(0)

# API 地址
api_url = "http://localhost:8000/predict"

# 发送请求
files = {'image': ('test.jpg', img_bytes, 'image/jpeg')}
data = {
    'instruction': 'pick up the red cup',
    'use_dummy_tactile': True
}

print("=" * 80)
print("测试 API 响应格式")
print("=" * 80)
print(f"API URL: {api_url}")
print(f"Instruction: {data['instruction']}")
print("=" * 80)

try:
    response = requests.post(api_url, files=files, data=data, timeout=30)
    
    print(f"\n响应状态码: {response.status_code}")
    print(f"响应头 Content-Type: {response.headers.get('Content-Type', 'N/A')}")
    
    if response.status_code == 200:
        result = response.json()
        
        print("\n" + "=" * 80)
        print("完整 JSON 响应:")
        print("=" * 80)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        print("\n" + "=" * 80)
        print("数据解析测试:")
        print("=" * 80)
        
        # 测试不同的解析方式
        if 'action' in result:
            action = result['action']
            print(f"✓ result['action'] = {action}")
            print(f"  类型: {type(action)}")
            if isinstance(action, list):
                action_arr = np.array(action)
                print(f"  转换为 numpy: {action_arr}")
                print(f"  形状: {action_arr.shape}")
                print(f"  数据类型: {action_arr.dtype}")
        
        if 'action_breakdown' in result:
            breakdown = result['action_breakdown']
            print(f"\n✓ result['action_breakdown'] = {json.dumps(breakdown, indent=2)}")
            
            # 尝试从 breakdown 重建 action
            if 'translation' in breakdown and 'rotation' in breakdown and 'gripper' in breakdown:
                trans = breakdown['translation']
                rot = breakdown['rotation']
                grip = breakdown['gripper']
                
                rebuilt_action = [
                    trans.get('dx', 0),
                    trans.get('dy', 0),
                    trans.get('dz', 0),
                    rot.get('droll', 0),
                    rot.get('dpitch', 0),
                    rot.get('dyaw', 0),
                    grip.get('value', 0)
                ]
                print(f"\n从 breakdown 重建的 action: {rebuilt_action}")
        
        # 检查是否有其他字段
        print(f"\n所有顶层键: {list(result.keys())}")
        
        # 模拟仿真环境的解析（可能的方式）
        print("\n" + "=" * 80)
        print("仿真环境可能的解析方式:")
        print("=" * 80)
        
        # 方式1: 直接取 action
        if 'action' in result:
            action1 = np.array(result['action'])
            print(f"方式1: np.array(result['action']) = {action1}")
        
        # 方式2: 从 action_breakdown 提取
        if 'action_breakdown' in result:
            bd = result['action_breakdown']
            try:
                action2 = np.array([
                    bd['translation']['dx'],
                    bd['translation']['dy'],
                    bd['translation']['dz'],
                    bd['rotation']['droll'],
                    bd['rotation']['dpitch'],
                    bd['rotation']['dyaw'],
                    bd['gripper']['value']
                ])
                print(f"方式2: 从 action_breakdown 提取 = {action2}")
            except KeyError as e:
                print(f"方式2: 失败 - 缺少键 {e}")
        
        # 方式3: 检查是否有嵌套结构
        if 'data' in result:
            print(f"方式3: result['data'] = {result['data']}")
        
    else:
        print(f"\n❌ 错误: {response.status_code}")
        print(response.text)
        
except requests.exceptions.RequestException as e:
    print(f"\n❌ 网络错误: {e}")
    print("请确保 API 服务正在运行: python serve_vtla_api.py --checkpoint ... --port 8000")
except Exception as e:
    import traceback
    print(f"\n❌ 错误: {e}")
    traceback.print_exc()

print("\n" + "=" * 80)
