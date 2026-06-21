from __future__ import annotations

import os
import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def fedavg_aggregate(state_dicts: List[Dict[str, torch.Tensor]], weights: List[int]) -> Dict[str, torch.Tensor]:
    """
    FedAvg: weighted average by number of local train samples.
    修复：正确处理整数类型参数（如位置编码的索引），只聚合浮点类型参数。
    """
    total = float(sum(weights)) if sum(weights) > 0 else 1.0
    
    # 获取参考 state_dict
    ref_sd = state_dicts[0]
    out = {}
    
    for key in ref_sd.keys():
        # 检查参数类型
        param_dtype = ref_sd[key].dtype
        
        # 只聚合浮点类型参数（可训练参数）
        if param_dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
            # 初始化聚合结果（使用 float32 进行计算）
            aggregated = torch.zeros_like(ref_sd[key], dtype=torch.float32)
            
            for sd, w in zip(state_dicts, weights):
                alpha = float(w) / total
                aggregated += sd[key].float() * alpha
            
            # 转回原始类型（如果需要）
            if param_dtype != torch.float32:
                out[key] = aggregated.to(param_dtype)
            else:
                out[key] = aggregated
        else:
            # 整数/布尔类型参数（通常是 buffer 或不训练的参数）：直接复制第一个客户端的
            # 这些参数在 FedAvg 中不应该被平均
            out[key] = ref_sd[key].clone()
    
    return out