"""
FastAPI Service for Nano-VTLA Model
提供 /predict API 供远程仿真容器调用
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import torch
import base64
import torch.nn as nn
from PIL import Image
import io
import numpy as np
import pickle

from bunny.model.builder import load_pretrained_model
from bunny.model.tactile_encoder import TactileTower
from bunny.constants import IMAGE_TOKEN_INDEX
import argparse
import os


# ========================================
# Configuration (可通过命令行参数覆盖)
# ========================================
CHECKPOINT_PATH = "./outputs/nano_vtla_baseline/checkpoint_step70000.pt"
ACTION_STATS_PATH = "./outputs/nano_vtla_baseline/action_mean_std.pkl"
MODEL_PATH = "BAAI/Bunny-v1_0-2B-zh"
MODEL_TYPE = "qwen1.5-1.8b"
DEVICE = "cuda"
USE_BF16 = True
PORT = 8000

# ========================================
# Isaac Sim 接口请求/响应模型 (JSON)
# ========================================
class IsaacPredictRequest(BaseModel):
    """Isaac Sim 身体脚本的预测请求 (POST /predict_isaac)"""
    image: str = Field(..., description="Base64 编码的 RGB 图像字符串")
    current_pose: List[float] = Field(
        ...,
        min_length=7,
        max_length=7,
        description="当前末端位姿 [x, y, z, qx, qy, qz, qw] (位置米, 四元数)"
    )
    text_prompt: str = Field(
        default="What action should the robot take?",
        description="语言指令"
    )


class IsaacPredictResponse(BaseModel):
    """Isaac Sim 预测响应: 相对位姿变换 EE_{i+1}T_{EE_i}"""
    success: bool
    delta_pose: Optional[List[float]] = Field(
        None,
        description="6-DoF 相对位姿 [dx, dy, dz, drx, dry, drz] (平移米, 旋转弧度)"
    )
    error: Optional[str] = None

# ========================================
# Global Model Storage
# ========================================
class VTLAModelService:
    """全局模型服务"""
    def __init__(self, checkpoint_path=None, action_stats_path=None, model_path=None, model_type=None, device=None, use_bf16=None):
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.action_mean = None
        self.action_std = None
        
        # 使用传入参数或全局默认值
        self.checkpoint_path = checkpoint_path or CHECKPOINT_PATH
        self.action_stats_path = action_stats_path or ACTION_STATS_PATH
        self.model_path = model_path or MODEL_PATH
        self.model_type = model_type or MODEL_TYPE
        self.device = device or DEVICE
        self.use_bf16 = use_bf16 if use_bf16 is not None else USE_BF16
        self.dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        
    def load_model(self):
        """加载模型和统计信息"""
        print("=" * 80)
        print("Loading Nano-VTLA Model")
        print("=" * 80)
        
        # Load base model
        print(f"[1/4] Loading base model: {self.model_path}")
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path=self.model_path,
            model_base=None,
            model_name=self.model_path,
            model_type=self.model_type,
            device=self.device
        )
        
        # Convert to BF16
        if self.use_bf16:
            self.model = self.model.to(torch.bfloat16)
            print(f"[2/4] Converted to BF16")
        
        # Add tactile tower
        self.model.model.config.use_tactile = True
        self.model.model.tactile_tower = TactileTower(
            pretrained=True,
            freeze_encoder=False,
            llm_hidden_size=self.model.config.hidden_size
        ).to(device=self.device, dtype=self.dtype)
        
        self.model.model.tactile_pos_embedding = nn.Parameter(
            torch.zeros(1, 1, self.model.config.hidden_size, device=self.device, dtype=self.dtype)
        )
        
        print(f"[3/4] Added tactile tower")
        
        # Load checkpoint
        print(f"[4/4] Loading checkpoint: {self.checkpoint_path}")
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location='cpu')
        self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"      Loaded checkpoint (step {ckpt.get('global_step', 0)})")
        
        # Load action normalization stats
        if os.path.exists(self.action_stats_path):
            with open(self.action_stats_path, 'rb') as f:
                stats = pickle.load(f)
                self.action_mean = torch.tensor(stats['mean'], dtype=torch.float32)
                self.action_std = torch.tensor(stats['std'], dtype=torch.float32)
            print(f"[Stats] Action normalization loaded")
            print(f"  Mean: {self.action_mean.numpy()}")
            print(f"  Std:  {self.action_std.numpy()}")
        else:
            print(f"[Warning] Action stats not found: {self.action_stats_path}, using identity normalization")
            # 使用单位归一化（不改变数值）
            self.action_mean = torch.zeros(7, dtype=torch.float32)
            self.action_std = torch.ones(7, dtype=torch.float32)
        
        # Set to eval mode
        self.model.eval()
        
        print("=" * 80)
        print("✅ Model Ready for Inference")
        print("=" * 80)
    
    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """反归一化动作（Z-score -> 物理单位）"""
        return action * self.action_std + self.action_mean
    
    @torch.no_grad()
    def predict(self, rgb_image: Image.Image, instruction: str, use_dummy_tactile: bool = True):
        """
        预测动作
        
        Args:
            rgb_image: PIL Image (任意尺寸)
            instruction: 语言指令
            use_dummy_tactile: 是否使用 dummy 触觉（黑色图像）
        
        Returns:
            action: (7,) numpy array [dx, dy, dz, droll, dpitch, dyaw, gripper]
                    单位: 米 (m) 和 弧度 (rad)
        """
        # 1. Preprocess RGB image (resize to 384x384)
        processed_image = self.image_processor([rgb_image], return_tensors='pt')['pixel_values']
        processed_image = processed_image.to(device=self.device, dtype=self.dtype)
        
        # 2. Generate tactile input
        if use_dummy_tactile:
            # Black image (128x128x3)
            tactile = torch.zeros(1, 3, 128, 128, device=self.device, dtype=self.dtype)
        else:
            # TODO: 如果有真实触觉，在这里处理
            tactile = torch.zeros(1, 3, 128, 128, device=self.device, dtype=self.dtype)
        
        # 3. Tokenize instruction with IMAGE_TOKEN_INDEX
        if not instruction:
            instruction = "What action should the robot take?"
        
        ids = [IMAGE_TOKEN_INDEX] + self.tokenizer(instruction).input_ids
        input_ids = torch.tensor([ids], dtype=torch.long).to(self.device)
        attention_mask = torch.ones_like(input_ids)
        
        # 4. Forward pass
        num_image_tokens = 729  # SigLIP 384x384
        
        with torch.amp.autocast('cuda', enabled=self.use_bf16, dtype=self.dtype):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=processed_image,
                tactile_images=tactile,
                num_image_tokens=num_image_tokens,
                use_cache=False,
                return_dict=True
            )
        
        # 5. Extract action prediction
        action_pred = outputs.action_prediction.squeeze(0).cpu()  # (7,)
        
        # 6. Denormalize (Z-score -> physical units)
        action_denorm = self.denormalize_action(action_pred)
        
        return action_denorm.numpy()


# Global model service (will be initialized in startup)
vtla_service = None

# ========================================
# FastAPI App
# ========================================
app = FastAPI(
    title="Nano-VTLA API",
    description="Vision-Tactile-Language-Action Model Inference Service",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    global vtla_service
    print("\n[FastAPI] Starting Nano-VTLA Service...")
    vtla_service.load_model()
    print("[FastAPI] Service ready!\n")


# 使用新的 lifespan 方式（兼容旧版本）
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan 事件处理器（替代 on_event）"""
    # Startup
    global vtla_service
    print("\n[FastAPI] Starting Nano-VTLA Service...")
    vtla_service.load_model()
    print("[FastAPI] Service ready!\n")
    yield
    # Shutdown (如果需要清理资源，在这里添加)

# 更新 app 初始化（如果支持）
# app = FastAPI(..., lifespan=lifespan)


@app.get("/")
async def root():
    """健康检查"""
    global vtla_service
    return {
        "service": "Nano-VTLA API",
        "status": "running",
        "model": vtla_service.model_path if vtla_service else MODEL_PATH,
        "checkpoint": vtla_service.checkpoint_path if vtla_service else CHECKPOINT_PATH,
        "device": vtla_service.device if vtla_service else DEVICE,
        "dtype": "BF16" if (vtla_service.use_bf16 if vtla_service else USE_BF16) else "FP16"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    is_loaded = vtla_service.model is not None
    return {
        "status": "healthy" if is_loaded else "loading",
        "model_loaded": is_loaded,
        "cuda_available": torch.cuda.is_available(),
        "gpu_memory": f"{torch.cuda.memory_allocated() / 1e9:.2f} GB" if torch.cuda.is_available() else "N/A"
    }


@app.post("/predict")
async def predict_action(
    image: UploadFile = File(..., description="RGB image (任意尺寸，会 resize 到 384x384)"),
    instruction: str = Form(default="What action should the robot take?", description="语言指令"),
    use_dummy_tactile: bool = Form(default=True, description="是否使用 dummy 触觉（黑色图像）")
):
    """
    预测 7-DoF 机器人动作
    
    输入:
        - image: RGB 图像文件
        - instruction: 语言指令
        - use_dummy_tactile: 是否使用 dummy 触觉
    
    输出:
        - action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        - units: 平移单位为米(m)，旋转单位为弧度(rad)
    """
    try:
        # Read image
        image_bytes = await image.read()
        rgb_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        print(f"\n[Request] Image: {rgb_image.size}, Instruction: '{instruction}', Dummy Tactile: {use_dummy_tactile}")
        
        # Predict
        action = vtla_service.predict(rgb_image, instruction, use_dummy_tactile)
        
        # Format response
        response = {
            "success": True,
            "action": action.tolist(),
            "action_breakdown": {
                "translation": {
                    "dx": float(action[0]),
                    "dy": float(action[1]),
                    "dz": float(action[2]),
                    "unit": "meters"
                },
                "rotation": {
                    "droll": float(action[3]),
                    "dpitch": float(action[4]),
                    "dyaw": float(action[5]),
                    "unit": "radians"
                },
                "gripper": {
                    "value": float(action[6]),
                    "range": "0 (open) to 1 (closed)"
                }
            },
            "metadata": {
                "instruction": instruction,
                "image_size": list(rgb_image.size),
                "used_dummy_tactile": use_dummy_tactile
            }
        }
        
        print(f"[Response] Action: {action}")
        
        return JSONResponse(content=response)
        
    except Exception as e:
        print(f"[Error] {str(e)}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "action": None
            }
        )


@app.post("/predict_isaac", response_model=IsaacPredictResponse)
async def predict_isaac(req: IsaacPredictRequest):
    """
    Isaac Sim 身体脚本专用接口 (JSON)。
    
    输入 (JSON):
        - image: Base64 编码的 RGB 图像
        - current_pose: [x, y, z, qx, qy, qz, qw] 当前末端位姿
        - text_prompt: 语言指令
    
    输出 (JSON):
        - delta_pose: [dx, dy, dz, drx, dry, drz] 相对位姿变换 EE_{i+1}T_{EE_i}
          (平移单位: 米, 旋转单位: 弧度, 欧拉角 droll/dpitch/dyaw)
    """
    try:
        # 1. Base64 解码 -> PIL Image（模型内部会通过 image_processor 缩放到 384x384）
        try:
            raw = base64.b64decode(req.image)
            rgb_image = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "delta_pose": None,
                    "error": f"Invalid base64 image: {e}"
                }
            )
        
        # 2. current_pose 仅作记录/预留；当前 VTLA 模型无 proprioception 分支，推理不修改
        _ = req.current_pose  # [x, y, z, qx, qy, qz, qw]
        
        # 3. 调用现有 VTLA 推理（复用原有逻辑，不修改）
        action = vtla_service.predict(
            rgb_image,
            req.text_prompt,
            use_dummy_tactile=True
        )
        # action: (7,) [dx, dy, dz, droll, dpitch, dyaw, gripper]
        
        # 4. 转为 6-DoF delta_pose: [dx, dy, dz, drx, dry, drz]（去掉 gripper）
        delta_pose = [
            float(action[0]),
            float(action[1]),
            float(action[2]),
            float(action[3]),
            float(action[4]),
            float(action[5]),
        ]
        
        print(
            f"[predict_isaac] image_size={rgb_image.size}, "
            f"current_pose=[...], text_prompt='{req.text_prompt[:50]}...', "
            f"delta_pose={delta_pose}"
        )
        
        return JSONResponse(
            content={
                "success": True,
                "delta_pose": delta_pose,
                "error": None
            }
        )
        
    except Exception as e:
        print(f"[predict_isaac Error] {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "delta_pose": None,
                "error": str(e)
            }
        )


@app.get("/stats")
async def get_action_stats():
    """获取动作归一化统计信息"""
    if vtla_service.action_mean is None:
        return {"error": "Stats not loaded"}
    
    return {
        "mean": vtla_service.action_mean.numpy().tolist(),
        "std": vtla_service.action_std.numpy().tolist(),
        "description": {
            "0-2": "translation (dx, dy, dz) in meters",
            "3-5": "rotation (droll, dpitch, dyaw) in radians",
            "6": "gripper (0=open, 1=closed)"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Nano-VTLA FastAPI Service")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH, help="Path to checkpoint file")
    parser.add_argument("--action-stats", type=str, default=ACTION_STATS_PATH, help="Path to action stats file")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="HuggingFace model path")
    parser.add_argument("--model-type", type=str, default=MODEL_TYPE, help="Model type")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device (cuda/cpu)")
    parser.add_argument("--bf16", action="store_true", default=USE_BF16, help="Use BF16 precision")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision (overrides --bf16)")
    parser.add_argument("--port", type=int, default=PORT, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    
    args = parser.parse_args()
    
    # 确定精度
    use_bf16 = args.bf16 and not args.fp16
    
    # 初始化全局服务（模块级别，不需要 global 声明）
    vtla_service = VTLAModelService(
        checkpoint_path=args.checkpoint,
        action_stats_path=args.action_stats,
        model_path=args.model,
        model_type=args.model_type,
        device=args.device,
        use_bf16=use_bf16
    )
    
    print("\n" + "=" * 80)
    print("Nano-VTLA FastAPI Service")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Dtype: {'BF16' if use_bf16 else 'FP16'}")
    print("=" * 80)
    print(f"\nStarting server on http://{args.host}:{args.port}")
    print(f"API 文档: http://{args.host}:{args.port}/docs")
    print("=" * 80 + "\n")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )
