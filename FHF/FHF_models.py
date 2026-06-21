from __future__ import annotations

import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 禁用 Flash Attention 警告，强制使用 Memory Efficient
warnings.filterwarnings("ignore", message=".*flash attention.*")
# 全局设置：禁用 Flash，启用 Memory Efficient
if hasattr(torch.backends.cuda, 'sdp_kernel'):
    # PyTorch 2.0+ 新 API
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)


class LSTM_reg(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size: int = 1,
        dropout=0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


class RevIN(nn.Module):
    """
    Reversible Instance Normalization
    Input/Output shape: [B, L, C]
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        subtract_last: bool = False,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

        self._cached_mean = None
        self._cached_stdev = None
        self._cached_last = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"RevIN expects [B, L, C], got {tuple(x.shape)}")

        if mode == "norm":
            if self.subtract_last:
                self._cached_last = x[:, -1:, :].detach()
                x_centered = x - self._cached_last
                self._cached_mean = None
            else:
                self._cached_mean = x.mean(dim=1, keepdim=True).detach()
                x_centered = x - self._cached_mean
                self._cached_last = None

            self._cached_stdev = torch.sqrt(
                torch.var(x_centered, dim=1, keepdim=True, unbiased=False) + self.eps
            ).detach()

            x = x_centered / self._cached_stdev

            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x

        if mode == "denorm":
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)

            x = x * self._cached_stdev

            if self.subtract_last:
                x = x + self._cached_last
            else:
                x = x + self._cached_mean
            return x

        raise ValueError(f"Unknown mode: {mode}")


# =========================================================
# 2. Positional Encoding
# =========================================================
class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, num_patches: int, d_model: int):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        return x + self.pos[:, : x.size(1), :]


# =========================================================
# 3. Flatten Head
# =========================================================
class FlattenHead(nn.Module):
    """
    Input : [B, C, N, D]
    Output: [B, pred_len, C]
    """

    def __init__(
        self,
        c_in: int,
        patch_num: int,
        d_model: int,
        pred_len: int,
        head_dropout: float = 0.0,
        individual: bool = False,
    ):
        super().__init__()
        self.c_in = c_in
        self.patch_num = patch_num
        self.d_model = d_model
        self.pred_len = pred_len
        self.individual = individual

        if individual:
            self.linears = nn.ModuleList(
                [nn.Linear(patch_num * d_model, pred_len) for _ in range(c_in)]
            )
            self.dropouts = nn.ModuleList(
                [nn.Dropout(head_dropout) for _ in range(c_in)]
            )
        else:
            self.linear = nn.Linear(patch_num * d_model, pred_len)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N, D]
        B, C, N, D = x.shape

        if self.individual:
            outs = []
            for i in range(C):
                z = x[:, i, :, :].reshape(B, N * D)    # [B, N*D]
                z = self.linears[i](z)                 # [B, pred_len]
                z = self.dropouts[i](z)
                outs.append(z.unsqueeze(-1))           # [B, pred_len, 1]
            out = torch.cat(outs, dim=-1)              # [B, pred_len, C]
        else:
            x = x.reshape(B, C, N * D)                 # [B, C, N*D]
            x = self.linear(x)                         # [B, C, pred_len]
            x = self.dropout(x)
            out = x.transpose(1, 2)                    # [B, pred_len, C]

        return out


# =========================================================
# 4. PatchTST Backbone
# =========================================================
class PatchTST(nn.Module):
    """
    Input : [B, seq_len, c_in]
    Output: [B, pred_len, c_in]
    """

    def __init__(
        self,
        c_in: int,
        seq_len: int,
        pred_len: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        fc_dropout: float = 0.1,
        head_dropout: float = 0.0,
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        individual_head: bool = False,
    ):
        super().__init__()

        if patch_len > seq_len:
            raise ValueError(f"patch_len={patch_len} cannot exceed seq_len={seq_len}")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.c_in = c_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.revin_enabled = revin

        # RevIN
        self.revin = RevIN(
            num_features=c_in,
            affine=affine,
            subtract_last=subtract_last,
        ) if revin else None

        # number of patches with end padding
        self.patch_num = math.ceil((seq_len - patch_len) / stride) + 1
        self.pad_len = max(0, (self.patch_num - 1) * stride + patch_len - seq_len)

        # patch embedding
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = LearnablePositionalEmbedding(self.patch_num, d_model)
        self.input_dropout = nn.Dropout(dropout)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        # official-like post norm
        self.norm = nn.BatchNorm1d(d_model)

        self.fc_dropout = nn.Dropout(fc_dropout)

        self.head = FlattenHead(
            c_in=c_in,
            patch_num=self.patch_num,
            d_model=d_model,
            pred_len=pred_len,
            head_dropout=head_dropout,
            individual=individual_head,
        )

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, seq_len, c_in]
        return: [B, c_in, patch_num, patch_len]
        """
        x = x.transpose(1, 2)  # [B, C, L]

        if self.pad_len > 0:
            x = F.pad(x, (0, self.pad_len), mode="replicate")

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # [B, C, patch_num, patch_len]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"PatchTST expects [B, seq_len, c_in], got {tuple(x.shape)}")
        if x.size(1) != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x.size(1)}")
        if x.size(2) != self.c_in:
            raise ValueError(f"Expected c_in={self.c_in}, got {x.size(2)}")

        # RevIN normalize
        if self.revin_enabled:
            x = self.revin(x, "norm")

        # patchify
        x = self._patchify(x)                      # [B, C, N, P]
        B, C, N, P = x.shape

        # channel-independence:
        # treat each variable independently, shared embedding/shared encoder
        x = x.reshape(B * C, N, P)                # [B*C, N, P]
        x = self.patch_embed(x)                   # [B*C, N, D]
        x = self.pos_embed(x)
        x = self.input_dropout(x)

        # 使用上下文管理器确保 Memory Efficient Attention
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=True):
                x = self.encoder(x)                       # [B*C, N, D]
        else:
            x = self.encoder(x)

        # batch norm on feature dimension
        x = x.transpose(1, 2)                     # [B*C, D, N]
        x = self.norm(x)
        x = x.transpose(1, 2)                     # [B*C, N, D]

        # restore channel dimension
        x = x.reshape(B, C, N, self.d_model)      # [B, C, N, D]
        x = self.fc_dropout(x)

        # head
        out = self.head(x)                        # [B, pred_len, C]

        # RevIN denormalize
        if self.revin_enabled:
            out = self.revin(out, "denorm")
        
        # if cin = 1, then make [B, pred_len, 1] -> [B, pred_len]
        if self.c_in == 1:
            out = out.squeeze(-1)

        return out


def build_model(args) -> nn.Module:
    if args.model_type == "lstm":
        return LSTM_reg(
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            output_size=args.fh + 1,
            dropout=args.dropout,
        )
    elif args.model_type == "patchtst":
        # 计算 patch_len 和 stride（确保是整数）
        patch_len = min(max(int(args.lags // 4), 4), 16)
        stride = min(max(int(args.lags // 8), 2), 8)
        
        return PatchTST(
            c_in=args.c_in,
            seq_len=args.lags,
            pred_len=args.fh + 1,
            patch_len=patch_len,
            stride=stride,
            d_model=args.d_model,
            n_heads=args.n_heads,
            e_layers=args.e_layers,
            d_ff=args.d_model * 4,
            dropout=args.dropout,
            fc_dropout=args.dropout,
            head_dropout=0.0,
            revin=True,
            affine=True,
            subtract_last=False,
            individual_head=False,
        )
    
    raise ValueError(f"Unsupported model_type: {args.model_type}")