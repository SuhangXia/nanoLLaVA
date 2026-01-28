"""
Tactile Encoder for Nano-VTLA
Based on TLA.pdf: Uses ResNet-18 pretrained on ImageNet to process 128x128 GelSight images
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class TactileEncoder(nn.Module):
    """
    ResNet-18 based tactile encoder for processing GelSight tactile images.
    
    Input: (B, 3, 128, 128) - GelSight RGB images
    Output: (B, 512) - Feature vector
    
    Architecture:
    - ResNet-18 backbone (pretrained on ImageNet)
    - Remove final classification layer
    - Output 512-dim feature vector from avgpool layer
    """
    
    def __init__(self, pretrained=True, freeze_backbone=False):
        super().__init__()
        
        # Load pretrained ResNet-18
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.resnet = resnet18(weights=weights)
        else:
            self.resnet = resnet18(weights=None)
        
        # Remove the final classification layer
        # ResNet-18 structure: conv layers -> avgpool -> fc
        # We keep everything except the fc layer
        self.backbone = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Freeze backbone if specified (for Stage 1 training)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.hidden_size = 512  # ResNet-18 feature dimension
    
    def forward(self, tactile_images):
        """
        Args:
            tactile_images: (B, 3, 128, 128) tensor
        
        Returns:
            features: (B, 512) tensor
        """
        # Forward through ResNet backbone
        features = self.backbone(tactile_images)  # (B, 512, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (B, 512)
        
        return features


class TactileProjector(nn.Module):
    """
    MLP projector to map tactile features to LLM embedding space.
    
    Input: (B, 512) - ResNet-18 features
    Output: (B, llm_hidden_size) - Projected features in LLM space
    
    Architecture: Linear -> GELU -> Linear
    Following TLA.pdf design
    """
    
    def __init__(self, tactile_hidden_size=512, llm_hidden_size=1024):
        super().__init__()
        
        self.projector = nn.Sequential(
            nn.Linear(tactile_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
    
    def forward(self, tactile_features):
        """
        Args:
            tactile_features: (B, 512) tensor
        
        Returns:
            projected_features: (B, llm_hidden_size) tensor
        """
        return self.projector(tactile_features)


class TactileTower(nn.Module):
    """
    Complete tactile processing tower: Encoder + Projector
    
    This module mirrors the Vision Tower structure in nanoLLaVA
    """
    
    def __init__(
        self, 
        pretrained=True, 
        freeze_encoder=False,
        llm_hidden_size=1024
    ):
        super().__init__()
        
        self.encoder = TactileEncoder(
            pretrained=pretrained, 
            freeze_backbone=freeze_encoder
        )
        self.projector = TactileProjector(
            tactile_hidden_size=self.encoder.hidden_size,
            llm_hidden_size=llm_hidden_size
        )
        
        self.hidden_size = llm_hidden_size
    
    def forward(self, tactile_images):
        """
        Args:
            tactile_images: (B, 3, 128, 128) tensor
        
        Returns:
            tactile_embeddings: (B, llm_hidden_size) tensor
        """
        # Encode tactile images
        features = self.encoder(tactile_images)  # (B, 512)
        
        # Project to LLM space
        embeddings = self.projector(features)  # (B, llm_hidden_size)
        
        return embeddings
