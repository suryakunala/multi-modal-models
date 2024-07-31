# Switch Transformer Implementation
# Imports
import torch
import torch.Tensor as Tensor
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

    def forward(self, x:Tensor):
        # capture shape
        seq_len, batch_size, d_model = x.shape
        # Flatten the input tensor
        x = x.view(-1, d_model)
        # get routing probabilities
        route_prob = self.softmax(self.switch(x))
        route_prob_max, route_indices = route_prob.max(dim=-1)
        # get indices of tokens going to each expert
        indices_list = [torch.eq(route_indices, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]
        # Initialize an empty tensor to store output
        output = torch.zeros(x.shape)
        # Calculate capacity of each expert
        # capacity_factor * (total_tokens/n_experts)
        capacity = int(self.capacity_factor * len(x) / self.n_experts)
        # Number of tokens routed to each expert.
        counts = x.new_tensor([len(indices_list[i]) for i in range(self.n_experts)])
        # Initialize an empty list of dropped tokens
        dropped = []
        # Only drop tokens if `drop_tokens` is `True`.
        if self.drop_tokens:
            # Drop tokens in each of the experts
            for i in range(self.n_experts):
                # Ignore if the expert is not over capacity
                if len(indices_list[i]) <= capacity:
                    continue
                # Shuffle indexes before dropping
                indices_list[i] = indices_list[i][torch.randperm(len(indices_list[i]))]
                # Collect the tokens over capacity as dropped tokens
                dropped.append(indices_list[i][capacity:])
                # Keep only the tokens upto the capacity of the expert
                indices_list[i] = indices_list[i][:capacity]

        # Get outputs of the expert FFNs
        # Expert forward
        expert_output = [self.experts[i](x[indices_list[i], :]) for i in range(self.n_experts)]
        # Assign to final output
        for i in range(self.n_experts):
            output[indices_list[i], :] = expert_output[i]
            # Pass through the dropped tokens
        if dropped:
            dropped = torch.cat(dropped)
            output[dropped, :] = x[dropped, :]

        if self.is_scale_prob:
            # Multiply by the expert outputs by the probabilities $y = p_i(x) E_i(x)$
            output = output * route_prob_max.view(-1, 1)
        output = output.view(seq_len, batch_size, d_model)
        # Return
        #
        # * the final output
        # * number of tokens routed to each expert
        # * sum of probabilities for each expert
        # * number of tokens dropped.
        # * routing probabilities of the selected experts
        #
        # These are used for the load balancing loss and logging
        return output, counts, route_prob.sum(0), len(dropped), route_prob_max

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
