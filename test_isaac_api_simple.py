"""
简化的 Isaac Sim API 测试脚本
不需要完整 Isaac Sim 场景，仅测试与 VTLA API 的通信
"""

import numpy as np
import requests
import base64
import io
from PIL import Image
import argparse


def test_vtla_api_connection(api_url="http://localhost:8000"):
    """测试 VTLA API 连接和响应"""
    
    print("=" * 80)
    print("测试 VTLA API 连接 (Isaac Sim 接口)")
    print("=" * 80)
    print(f"API 地址: {api_url}")
    print("=" * 80)
    
    # 1. 健康检查
    print("\n[1/3] 健康检查...")
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"  ✅ API 健康")
            print(f"     状态: {health.get('status', 'unknown')}")
            print(f"     模型已加载: {health.get('model_loaded', False)}")
            print(f"     CUDA 可用: {health.get('cuda_available', False)}")
            print(f"     GPU 内存: {health.get('gpu_memory', 'N/A')}")
        else:
            print(f"  ❌ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ 无法连接: {e}")
        return False
    
    # 2. 测试 Isaac Sim 接口 (POST /predict_isaac)
    print("\n[2/3] 测试 Isaac Sim 接口 (POST /predict_isaac)...")
    
    # 创建测试图像（384x384 红色）
    test_image = Image.new('RGB', (384, 384), color=(255, 0, 0))
    img_bytes = io.BytesIO()
    test_image.save(img_bytes, format='JPEG')
    img_b64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    
    # 测试位姿（单位矩阵位置 + 单位四元数）
    current_pose = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]  # [x, y, z, qx, qy, qz, qw]
    
    # 测试指令
    text_prompt = "pick up the red cube"
    
    payload = {
        "image": img_b64,
        "current_pose": current_pose,
        "text_prompt": text_prompt
    }
    
    try:
        response = requests.post(
            f"{api_url}/predict_isaac",
            json=payload,
            timeout=30
        )
        
        print(f"  HTTP 状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"  ✅ 请求成功")
            print(f"     success: {result.get('success', False)}")
            
            if result.get('success', False):
                delta_pose = result.get('delta_pose', None)
                if delta_pose:
                    print(f"     delta_pose: {delta_pose}")
                    print(f"       - 平移 (dx, dy, dz): [{delta_pose[0]:.6f}, {delta_pose[1]:.6f}, {delta_pose[2]:.6f}] m")
                    print(f"       - 旋转 (drx, dry, drz): [{delta_pose[3]:.6f}, {delta_pose[4]:.6f}, {delta_pose[5]:.6f}] rad")
                    print(f"       - 平移距离: {np.linalg.norm(delta_pose[:3]):.6f} m")
                    print(f"       - 旋转幅度: {np.linalg.norm(delta_pose[3:]):.6f} rad ({np.degrees(np.linalg.norm(delta_pose[3:])):.2f}°)")
                else:
                    print(f"  ⚠️  响应中没有 delta_pose")
            else:
                print(f"  ❌ API 返回失败: {result.get('error', 'Unknown')}")
                return False
        else:
            print(f"  ❌ HTTP 错误: {response.text}")
            return False
            
    except Exception as e:
        print(f"  ❌ 请求失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 多次测试（验证稳定性）
    print("\n[3/3] 多次测试（验证稳定性）...")
    num_tests = 5
    success_count = 0
    
    for i in range(num_tests):
        try:
            response = requests.post(
                f"{api_url}/predict_isaac",
                json=payload,
                timeout=30
            )
            if response.status_code == 200 and response.json().get('success', False):
                success_count += 1
                delta = response.json()['delta_pose']
                print(f"  测试 {i+1}/{num_tests}: ✓ delta_pose={[f'{x:.4f}' for x in delta[:3]]}")
            else:
                print(f"  测试 {i+1}/{num_tests}: ✗")
        except Exception as e:
            print(f"  测试 {i+1}/{num_tests}: ✗ {e}")
    
    print(f"\n  成功率: {success_count}/{num_tests} ({100*success_count/num_tests:.1f}%)")
    
    # 总结
    print("\n" + "=" * 80)
    if success_count == num_tests:
        print("✅ 所有测试通过！Isaac Sim 接口工作正常")
        print("=" * 80)
        print("\n下一步：运行完整的 Isaac Sim 推理脚本")
        print("  python isaac_sim_vtla_inference.py --api-url http://localhost:8000")
        return True
    else:
        print(f"⚠️  部分测试失败 ({success_count}/{num_tests})")
        print("=" * 80)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试 VTLA Isaac Sim API")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000",
                       help="VTLA API 服务地址")
    
    args = parser.parse_args()
    
    success = test_vtla_api_connection(api_url=args.api_url)
    
    if not success:
        print("\n建议:")
        print("1. 确保 VTLA API 服务正在运行")
        print("2. 检查 API 地址是否正确")
        print("3. 查看 API 服务日志")
