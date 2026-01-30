# Nano-VTLA API 数据格式说明

## API 端点

**POST** `/predict`

## 请求格式

### Content-Type
`multipart/form-data`

### 参数
- `image` (File, required): RGB 图像文件（任意尺寸，会自动 resize 到 384x384）
- `instruction` (String, optional): 语言指令，默认 "What action should the robot take?"
- `use_dummy_tactile` (Boolean, optional): 是否使用 dummy 触觉（黑色图像），默认 `true`

### 示例请求（Python）
```python
import requests

files = {'image': ('image.jpg', open('image.jpg', 'rb'), 'image/jpeg')}
data = {
    'instruction': 'pick up the red cup',
    'use_dummy_tactile': True
}
response = requests.post('http://localhost:8000/predict', files=files, data=data)
```

## 响应格式

### 成功响应 (200 OK)

```json
{
  "success": true,
  "action": [dx, dy, dz, droll, dpitch, dyaw, gripper],
  "action_breakdown": {
    "translation": {
      "dx": 0.001234,
      "dy": 0.002345,
      "dz": -0.000567,
      "unit": "meters"
    },
    "rotation": {
      "droll": 0.001234,
      "dpitch": -0.002345,
      "dyaw": 0.000567,
      "unit": "radians"
    },
    "gripper": {
      "value": 0.103456,
      "range": "0 (open) to 1 (closed)"
    }
  },
  "metadata": {
    "instruction": "pick up the red cup",
    "image_size": [384, 384],
    "used_dummy_tactile": true
  }
}
```

### 错误响应 (500 Internal Server Error)

```json
{
  "success": false,
  "error": "错误信息",
  "action": null
}
```

## 数据解析

### 方式1: 直接使用 `action` 数组（推荐）

```python
import requests
import numpy as np

response = requests.post('http://localhost:8000/predict', files=files, data=data)
result = response.json()

if result['success']:
    # 直接获取 7-DoF 动作数组
    action = np.array(result['action'])  # shape: (7,)
    
    # 分解
    dx, dy, dz = action[0], action[1], action[2]
    droll, dpitch, dyaw = action[3], action[4], action[5]
    gripper = action[6]
    
    print(f"Translation: [{dx:.6f}, {dy:.6f}, {dz:.6f}] m")
    print(f"Rotation: [{droll:.6f}, {dpitch:.6f}, {dyaw:.6f}] rad")
    print(f"Gripper: {gripper:.4f}")
else:
    print(f"Error: {result['error']}")
```

### 方式2: 从 `action_breakdown` 提取

```python
if result['success']:
    breakdown = result['action_breakdown']
    
    dx = breakdown['translation']['dx']
    dy = breakdown['translation']['dy']
    dz = breakdown['translation']['dz']
    
    droll = breakdown['rotation']['droll']
    dpitch = breakdown['rotation']['dpitch']
    dyaw = breakdown['rotation']['dyaw']
    
    gripper = breakdown['gripper']['value']
    
    action = np.array([dx, dy, dz, droll, dpitch, dyaw, gripper])
```

## 数据单位

- **平移 (dx, dy, dz)**: 米 (meters)
- **旋转 (droll, dpitch, dyaw)**: 弧度 (radians)
- **夹爪 (gripper)**: 0.0 (打开) 到 1.0 (闭合)

## 常见问题

### Q: 为什么收到的 action 都是 [0, 0, 0, 0, 0, 0, 0]？

**A:** 可能的原因：
1. **解析路径错误**: 确保使用 `result['action']` 而不是其他路径
2. **数据类型错误**: 确保将 list 转换为 numpy array: `np.array(result['action'])`
3. **API 服务未正确加载模型**: 检查 API 日志，确认模型加载成功
4. **图像格式问题**: 确保图像是有效的 RGB 图像

### Q: 如何验证 API 返回的数据？

**A:** 运行测试脚本：
```bash
python test_api_response_format.py
```

### Q: 如何检查 API 健康状态？

**A:** 
```bash
curl http://localhost:8000/health
```

或访问：
```
http://localhost:8000/docs
```

## 完整示例代码

```python
import requests
import numpy as np
from PIL import Image
import io

def get_action_from_api(image, instruction, api_url="http://localhost:8000"):
    """
    从 API 获取动作预测
    
    Args:
        image: PIL Image 或 numpy array (H, W, 3)
        instruction: str, 语言指令
        api_url: str, API 基础 URL
    
    Returns:
        action: numpy array, shape (7,), [dx, dy, dz, droll, dpitch, dyaw, gripper]
    """
    # 转换图像为 PIL Image
    if isinstance(image, np.ndarray):
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
    
    # 准备请求
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    files = {'image': ('image.jpg', img_bytes, 'image/jpeg')}
    data = {
        'instruction': instruction if instruction else "What action should the robot take?",
        'use_dummy_tactile': True
    }
    
    # 发送请求
    try:
        response = requests.post(f"{api_url}/predict", files=files, data=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        if not result.get('success', False):
            raise ValueError(f"API returned error: {result.get('error', 'Unknown error')}")
        
        # 解析动作
        action = np.array(result['action'], dtype=np.float32)
        
        if len(action) != 7:
            raise ValueError(f"Expected 7-DoF action, got {len(action)} values")
        
        return action
        
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to connect to API: {e}")
    except KeyError as e:
        raise ValueError(f"Invalid API response format: missing key {e}")

# 使用示例
if __name__ == "__main__":
    # 创建测试图像
    test_image = Image.new('RGB', (384, 384), color='red')
    
    # 获取动作
    action = get_action_from_api(test_image, "pick up the red cup")
    
    print(f"Action: {action}")
    print(f"Translation: {action[:3]} m")
    print(f"Rotation: {action[3:6]} rad")
    print(f"Gripper: {action[6]}")
```
