"""
GPT2
---

Param Count: 117M
Layers: 12
Embedding Size: 768
Attention Heads: 12

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPT2Config:

