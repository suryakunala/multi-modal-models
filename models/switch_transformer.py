# Switch Transformer Implementation
# Imports
import torch.nn as nn
import copy


class Experts(nn.Module):
    def __init__(self,
                 n_experts: int,
                 expert_depth: int,
                 ffn_dim: int,
                 capacity_factor: float,
                 drop_tokens: bool,
                 is_scale_prob: bool
                 ):
        super().__init__()
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        self.is_scale_prob = is_scale_prob
        # FFNs
        expert = nn.ModuleList(
            nn.Linear(ffn_dim, ffn_dim) for _ in range(expert_depth)
        )
        # Experts
        self.experts = [copy.deepcopy(expert) for _ in range(self.n_experts)]

        # Routing layer
        self.switch = nn.Linear(ffn_dim, self.n_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self):
        ...


class SwitchTransformerLayer:
    def __init__(self):
        ...

    def forward(self):
        ...


class SwitchTransformer:
    def __int__(self):
        ...

    def forward(self):
        ...
