"""
PatchTST Model Components
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


# ============================================================================
# REVERSIBLE INSTANCE NORMALIZATION
# ============================================================================
class RevIN(nn.Module):
    """
    Reversible Instance Normalization for handling non-stationarity.
    
    Reference: Kim et al. "Reversible Instance Normalization for Accurate 
               Time-Series Forecasting against Distribution Shift" (ICLR 2022)
    """
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x: torch.Tensor, mode: str = 'norm') -> torch.Tensor:
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        return x
    
    def _get_statistics(self, x: torch.Tensor):
        dim2reduce = tuple(range(1, x.ndim - 1)) if x.ndim > 2 else 1
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x
    
    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev + self.mean
        return x


# ============================================================================
# PATCHING LAYER
# ============================================================================
class Patching(nn.Module):
    """Converts time series into patches using sliding window."""
    
    def __init__(self, patch_size: int, stride: int):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len]
        Returns:
            patches: [batch, num_patches, patch_size]
        """
        patches = x.unfold(dimension=1, size=self.patch_size, step=self.stride)
        return patches
    
    def get_num_patches(self, seq_len: int) -> int:
        return (seq_len - self.patch_size) // self.stride + 1


# ============================================================================
# POSITIONAL ENCODING
# ============================================================================
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# MASKING MECHANISM
# ============================================================================
class PatchMasking(nn.Module):
    """Randomly masks patches for self-supervised learning."""
    
    def __init__(self, mask_ratio: float = 0.4, mask_type: str = 'learnable', patch_size: int = None):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        
        if mask_type == 'learnable':
            if patch_size is None:
                raise ValueError("patch_size must be provided for learnable mask token")
            self.mask_token = nn.Parameter(torch.randn(1, 1, patch_size) * 0.02)
    
    def forward(
        self, 
        patches: torch.Tensor, 
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patches: [batch, num_patches, patch_size]
            training: If True, apply masking
        Returns:
            masked_patches: [batch, num_patches, patch_size]
            mask: [batch, num_patches] boolean mask
        """
        if not training:
            return patches, torch.zeros(
                patches.shape[:2], dtype=torch.bool, device=patches.device
            )
        
        batch_size, num_patches, patch_size = patches.shape
        
        # Create random mask
        mask = torch.rand(batch_size, num_patches, device=patches.device) < self.mask_ratio
        
        # Clone to avoid in-place modification
        masked_patches = patches.clone()
        
        # Apply masking
        if self.mask_type == 'zero':
            masked_patches[mask] = 0.0
        elif self.mask_type == 'learnable':
            num_masked = mask.sum().item()
            if num_masked > 0:
                masked_patches[mask] = self.mask_token.squeeze(0).squeeze(0).expand(
                    num_masked, -1
                )
        
        return masked_patches, mask


# ============================================================================
# PATCHTST MLM MODEL
# ============================================================================
class PatchTST_MLM(nn.Module):
    """
    PatchTST with Masked Language Modeling for self-supervised pre-training.
    """
    
    def __init__(
        self,
        seq_len: int,
        patch_size: int = 4,
        stride: int = 2,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        mask_ratio: float = 0.4,
        mask_type: str = 'learnable',
        use_revin: bool = True
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model
        self.use_revin = use_revin
        
        # Components
        if use_revin:
            self.revin = RevIN(num_features=1, affine=True)
        
        self.patching = Patching(patch_size=patch_size, stride=stride)
        self.num_patches = self.patching.get_num_patches(seq_len)
        
        self.masking = PatchMasking(mask_ratio=mask_ratio, mask_type=mask_type, patch_size=patch_size)
        
        self.patch_embedding = nn.Linear(patch_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.num_patches, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.reconstruction_head = nn.Linear(d_model, patch_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_embeddings: bool = False,
        training: bool = True
    ) -> dict:
        """
        Args:
            x: [batch, seq_len]
            return_embeddings: If True, return embeddings
            training: If True, apply masking
        Returns:
            Dictionary with reconstruction, mask, and optionally embeddings
        """
        # RevIN normalization
        if self.use_revin:
            x = x.unsqueeze(-1)
            x = self.revin(x, mode='norm')
            x = x.squeeze(-1)
        
        # Patching
        patches = self.patching(x)
        original_patches = patches.clone()
        
        # Masking
        masked_patches, mask = self.masking(patches, training=training)
        
        # Patch embedding
        patch_emb = self.patch_embedding(masked_patches)
        
        # Positional encoding
        patch_emb = self.pos_encoder(patch_emb)
        
        # Transformer encoding
        encoded = self.transformer_encoder(patch_emb)
        
        # Reconstruction
        reconstructed = self.reconstruction_head(encoded)
        
        output = {
            'reconstruction': reconstructed,
            'original_patches': original_patches,
            'mask': mask,
        }
        
        if return_embeddings:
            pooled_embedding = encoded.mean(dim=1)
            output['embeddings'] = encoded
            output['pooled_embedding'] = pooled_embedding
        
        return output
    
    def extract_embeddings(self, x: torch.Tensor, pooling: str = 'mean') -> torch.Tensor:
        """
        Extract embeddings without masking.
        
        Args:
            x: [batch, seq_len]
            pooling: 'mean', 'max', 'last', or 'cls'
        Returns:
            embeddings: [batch, d_model]
        """
        with torch.no_grad():
            output = self.forward(x, return_embeddings=True, training=False)
            encoded = output['embeddings']
            
            if pooling == 'mean':
                return encoded.mean(dim=1)
            elif pooling == 'max':
                return encoded.max(dim=1)[0]
            elif pooling == 'last':
                return encoded[:, -1, :]
            elif pooling == 'cls':
                return encoded[:, 0, :]
            else:
                raise ValueError(f"Unknown pooling: {pooling}")
    
    def compute_loss(self, output: dict) -> torch.Tensor:
        """Compute MSE loss on masked patches only."""
        reconstruction = output['reconstruction']
        original_patches = output['original_patches']
        mask = output['mask']
        
        if mask.sum() > 0:
            loss = F.mse_loss(
                reconstruction[mask],
                original_patches[mask],
                reduction='mean'
            )
        else:
            loss = torch.tensor(0.0, device=reconstruction.device)
        
        return loss
