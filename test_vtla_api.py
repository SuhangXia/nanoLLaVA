"""
测试 Nano-VTLA FastAPI 服务
"""

import requests
import numpy as np
from PIL import Image
import io

# API 配置
API_URL = "http://localhost:8000"

def test_health_check():
    """测试健康检查"""
    print("\n" + "=" * 60)
    print("测试健康检查")
    print("=" * 60)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    return response.json()


def test_predict_with_image(image_path: str, instruction: str = None):
    """测试预测 API"""
    print("\n" + "=" * 60)
    print("测试动作预测")
    print("=" * 60)
    
    # Load test image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    # Prepare request
    files = {'image': ('test.png', image_bytes, 'image/png')}
    data = {
        'instruction': instruction if instruction else "What action should the robot take?",
        'use_dummy_tactile': True
    }
    
    print(f"Sending request...")
    print(f"  Image: {image_path}")
    print(f"  Instruction: {data['instruction']}")
    print(f"  Use dummy tactile: {data['use_dummy_tactile']}")
    
    # Send request
    response = requests.post(f"{API_URL}/predict", files=files, data=data)
    
    print(f"\nResponse (Status {response.status_code}):")
    result = response.json()
    
    if result['success']:
        action = np.array(result['action'])
        print(f"\n预测动作:")
        print(f"  Translation (dx, dy, dz): {action[:3]} meters")
        print(f"  Rotation (droll, dpitch, dyaw): {action[3:6]} radians")
        print(f"  Rotation (degrees): {np.rad2deg(action[3:6])}")
        print(f"  Gripper: {action[6]:.4f} (0=open, 1=closed)")
        
        print(f"\n完整响应:")
        import json
        print(json.dumps(result, indent=2))
    else:
        print(f"❌ 错误: {result.get('error', 'Unknown error')}")
    
    return result


def test_get_stats():
    """测试获取统计信息"""
    print("\n" + "=" * 60)
    print("获取动作归一化统计")
    print("=" * 60)
    
    response = requests.get(f"{API_URL}/stats")
    stats = response.json()
    
    print(f"Mean: {stats['mean']}")
    print(f"Std:  {stats['std']}")
    print(f"\nDescription:")
    for k, v in stats['description'].items():
        print(f"  {k}: {v}")
    
    return stats


def create_dummy_image(size=(640, 480)):
    """创建测试图像"""
    img = Image.new('RGB', size, color=(100, 150, 200))
    
    # Add some simple patterns
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 300, 300], outline=(255, 0, 0), width=5)
    draw.ellipse([350, 200, 500, 350], fill=(0, 255, 0))
    
    return img


def main():
    print("=" * 80)
    print("Nano-VTLA API 测试客户端")
    print("=" * 80)
    
    # Test 1: Health check
    health = test_health_check()
    
    if not health.get('model_loaded', False):
        print("\n❌ 模型未加载，请先启动 serve_vtla_api.py")
        return
    
    # Test 2: Get stats
    stats = test_get_stats()
    
    # Test 3: Predict with dummy image
    print("\n" + "=" * 60)
    print("创建测试图像并预测")
    print("=" * 60)
    
    # Create and save test image
    test_image = create_dummy_image()
    test_image.save('/tmp/test_vtla_image.png')
    print(f"✅ 测试图像保存到: /tmp/test_vtla_image.png")
    
    # Test prediction
    result = test_predict_with_image(
        '/tmp/test_vtla_image.png',
        instruction="Pick up the red cube"
    )
    
    print("\n" + "=" * 80)
    print("✅ 所有测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
