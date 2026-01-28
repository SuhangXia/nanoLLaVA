"""
VTLA (Vision-Tactile-Language-Action) Architecture
Extended from bunny_arch.py with tactile modality support

Token Fusion Order (following TLA.pdf):
[Instruction_Tokens, Vision_Tokens, Tactile_Tokens]

This ensures language acts as global context conditioning the fusion of vision and tactile.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector
from .tactile_encoder import TactileTower

from bunny.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX

# Define a new token index for tactile data
TACTILE_TOKEN_INDEX = -300


class ActionHead(nn.Module):
    """
    3-layer MLP that maps hidden state to 7D action (dx, dy, dz, dr, dp, dyaw, gripper).
    Based on ViTaMIn-B.pdf relative action representation.
    """

    def __init__(self, hidden_size: int, action_dim: int = 7, intermediate_size: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, 256),
            nn.GELU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, hidden_size) - Last token hidden state
        
        Returns:
            action: (B, 7) - Predicted action [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        return self.mlp(x)


class ReasoningHead(nn.Module):
    """
    Placeholder for future Octopi-style reasoning head (Material/Hardness prediction).
    Currently returns None, but hooks are in place for future extensions.
    """
    
    def __init__(self, hidden_size: int, num_properties: int = 3):
        super().__init__()
        # Placeholder MLP for future material/hardness prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, num_properties)
        )
        self.enabled = False  # Disabled by default
    
    def forward(self, x: torch.Tensor):
        if not self.enabled:
            return None
        return self.mlp(x)


class VTLAMetaModel:
    """
    VTLA Meta Model with Vision, Tactile, and Language modalities.
    """

    def __init__(self, config):
        super(VTLAMetaModel, self).__init__(config)

        # Vision Tower (SigLIP from nanoLLaVA)
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
        
        # Tactile Tower (ResNet-18 + Projector)
        if hasattr(config, "use_tactile") and config.use_tactile:
            llm_hidden_size = getattr(config, "hidden_size", 1024)
            self.tactile_tower = TactileTower(
                pretrained=True,
                freeze_encoder=getattr(config, "freeze_tactile_encoder", False),
                llm_hidden_size=llm_hidden_size
            )
            
            # Learned positional embedding for tactile tokens
            # Following TLA.pdf: add positional encoding to distinguish tactile from vision
            self.tactile_pos_embedding = nn.Parameter(
                torch.zeros(1, 1, llm_hidden_size)
            )
            nn.init.normal_(self.tactile_pos_embedding, std=0.02)
        
        # Action Head (7-DoF relative action)
        if hasattr(config, "use_action_head") and config.use_action_head:
            llm_hidden_size = getattr(config, "hidden_size", 1024)
            self.action_head = ActionHead(
                hidden_size=llm_hidden_size,
                action_dim=7
            )
        
        # Reasoning Head (placeholder for future Octopi logic)
        if hasattr(config, "use_reasoning_head") and config.use_reasoning_head:
            llm_hidden_size = getattr(config, "hidden_size", 1024)
            self.reasoning_head = ReasoningHead(hidden_size=llm_hidden_size)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def get_tactile_tower(self):
        return getattr(self, 'tactile_tower', None)

    def initialize_vision_modules(self, model_args):
        """Initialize vision tower and projector"""
        vision_tower = model_args.vision_tower
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            self.vision_tower = vision_tower
        else:
            vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type')
        self.config.mm_hidden_size = vision_tower.hidden_size

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


class VTLAMetaForCausalLM(ABC):
    """
    VTLA Meta Class for Causal Language Modeling with multimodal inputs.
    """

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def get_tactile_tower(self):
        return self.get_model().get_tactile_tower()

    def encode_images(self, images):
        """Encode vision images to embeddings"""
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features
    
    def encode_tactile(self, tactile_images):
        """Encode tactile images to embeddings"""
        tactile_tower = self.get_tactile_tower()
        if tactile_tower is None:
            return None
        
        # Encode tactile: (B, 3, 128, 128) -> (B, llm_hidden_size)
        tactile_features = tactile_tower(tactile_images)  # (B, hidden_size)
        
        # Add positional embedding
        tactile_pos = self.get_model().tactile_pos_embedding  # (1, 1, hidden_size)
        tactile_features = tactile_features.unsqueeze(1) + tactile_pos  # (B, 1, hidden_size)
        
        return tactile_features

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, position_ids, attention_mask, past_key_values, labels, 
            images, tactile_images=None
    ):
        """
        Prepare multimodal inputs with Vision, Tactile, and Language.
        
        Token Fusion Order: [Instruction_Tokens, Vision_Tokens, Tactile_Tokens]
        
        This follows TLA.pdf: Language provides global context for vision-tactile fusion.
        """
        vision_tower = self.get_vision_tower()
        tactile_tower = self.get_tactile_tower()
        
        # Handle no multimodal case
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # Encode vision
        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1).to(self.device) for x in image_features]
        else:
            image_features = self.encode_images(images).to(self.device)
        
        # Encode tactile (if available)
        tactile_features = None
        if tactile_tower is not None and tactile_images is not None:
            if type(tactile_images) is list:
                concat_tactile = torch.cat([t for t in tactile_images], dim=0)
                tactile_features = self.encode_tactile(concat_tactile)
                split_sizes = [t.shape[0] for t in tactile_images]
                tactile_features = torch.split(tactile_features, split_sizes, dim=0)
                tactile_features = [x.to(self.device) for x in tactile_features]
            else:
                tactile_features = self.encode_tactile(tactile_images).to(self.device)

        # Setup dummy tensors
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # Remove padding
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in
                     zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            
            if num_images == 0:
                # No image tokens, just concat language + vision + tactile
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            # Split by image tokens
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [
                cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            # Token Fusion: [Instruction, Vision, Tactile]
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    # Add vision tokens
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device,
                                   dtype=cur_labels.dtype))
                    
                    # Add tactile tokens (after vision, following TLA.pdf design)
                    if tactile_features is not None:
                        if type(tactile_features) is list:
                            cur_tactile_features = tactile_features[batch_idx]
                        else:
                            cur_tactile_features = tactile_features[batch_idx:batch_idx+1]
                        
                        cur_new_input_embeds.append(cur_tactile_features)
                        cur_new_labels.append(
                            torch.full((cur_tactile_features.shape[0],), IGNORE_INDEX, device=cur_labels.device,
                                       dtype=cur_labels.dtype))

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Pad to max length
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype,
                                       device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                              device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                             device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
