# Isaac Sim 接口说明 (POST /predict_isaac)

与外部 Isaac Sim「身体」脚本通讯的 JSON 接口，复用现有 VTLA 推理逻辑，不修改原有 `/predict`（RLBench 等仍使用 multipart 上传）。

## 端点

- **URL**: `POST http://<host>:8000/predict_isaac`
- **Content-Type**: `application/json`
- **Host**: 服务默认监听 `0.0.0.0:8000`，便于 Docker 外网访问

## 请求格式 (JSON)

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `image` | string | 是 | Base64 编码的 RGB 图像字符串 |
| `current_pose` | list[float] | 是 | 长度 7：`[x, y, z, qx, qy, qz, qw]`（位置米，四元数） |
| `text_prompt` | string | 否 | 语言指令，默认 `"What action should the robot take?"` |

### 处理流程

1. **图像**：Base64 解码为 RGB 图像，由模型内部 `image_processor` 缩放到 **384x384**（SigLIP 输入尺寸）。
2. **current_pose**：当前仅作记录/预留；现有 VTLA 模型无 proprioception 分支，推理不依赖该字段。
3. **推理**：调用现有 VTLA 模型，得到 7-DoF 动作 `[dx, dy, dz, droll, dpitch, dyaw, gripper]`。
4. **输出**：取前 6 维作为相对位姿 `delta_pose`，即 `[dx, dy, dz, drx, dry, drz]`（`drx,dry,drz` 对应 droll, dpitch, dyaw，弧度）。

## 响应格式 (JSON)

### 成功 (200)

```json
{
  "success": true,
  "delta_pose": [dx, dy, dz, drx, dry, drz],
  "error": null
}
```

- **delta_pose**: 6-DoF 相对位姿变换 \(EE_{i+1}T_{EE_i}\)
  - `[0:3]`: 平移 (米) `dx, dy, dz`
  - `[3:6]`: 旋转 (弧度) `drx, dry, drz`（欧拉角 droll, dpitch, dyaw）

### 失败 (4xx/5xx)

```json
{
  "success": false,
  "delta_pose": null,
  "error": "错误信息"
}
```

## 调用示例

### Python

```python
import requests
import base64
import json

def predict_isaac(image_bytes: bytes, current_pose: list, text_prompt: str, api_url="http://localhost:8000"):
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "image": image_b64,
        "current_pose": current_pose,  # [x, y, z, qx, qy, qz, qw]
        "text_prompt": text_prompt
    }
    r = requests.post(f"{api_url}/predict_isaac", json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data["success"]:
        raise RuntimeError(data.get("error", "Unknown error"))
    return data["delta_pose"]  # [dx, dy, dz, drx, dry, drz]
```

### cURL

```bash
# 将图片转为 base64 后放入 JSON
IMAGE_B64=$(base64 -w0 your_image.png)
curl -X POST http://localhost:8000/predict_isaac \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$IMAGE_B64\", \"current_pose\": [0,0,0,0,0,0,1], \"text_prompt\": \"pick up the red cup\"}"
```

## 与 RLBench 接口的区别

| 项目 | RLBench (`POST /predict`) | Isaac Sim (`POST /predict_isaac`) |
|------|---------------------------|-----------------------------------|
| 请求方式 | multipart/form-data (File + Form) | JSON |
| 图像 | 文件上传 | Base64 字符串 |
| 位姿 | 无 | `current_pose` [x,y,z,qx,qy,qz,qw] |
| 输出 | 7-DoF `action` (含 gripper) | 6-DoF `delta_pose` (仅位姿) |

原有 `/predict` 接口未做任何修改，RLBench 等脚本可继续使用。
