import torch
import torch.nn as nn
import torch.nn.functional as F

class DelayLayerLogic(nn.Module):
    def __init__(self, in_features, out_features, max_delay):
        super(DelayLayerLogic, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_delay = max_delay

        # Weights & Delays
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        # self.delay = nn.Parameter(torch.rand(out_features, in_features) * max_delay)
        self.delay = nn.Parameter(torch.zeros(out_features, in_features))

    def forward(self, x):
        """
        x shape: (T, B, C)  where C = in_features
        Output shape: (T, B, out_features)
        """
        T, B, C = x.shape
        device = x.device

        # Clamp delays
        delay = torch.clamp(self.delay, 0.0, self.max_delay)  # (Out, In)

        floor_d = torch.floor(delay).long()                    # (Out, In)
        ceil_d = torch.ceil(delay).long()
        alpha = delay - floor_d.float()                        # (Out, In)

        # Time indices
        t = torch.arange(T, device=device).view(T, 1, 1, 1)   # (T, 1, 1, 1)

        # Compute delayed indices
        t_floor = torch.clamp(t - floor_d.view(1, 1, *floor_d.shape), 0, T - 1)
        t_ceil = torch.clamp(t - ceil_d.view(1, 1, *ceil_d.shape), 0, T - 1)

        # Gather with advanced indexing
        # Expand x for broadcasting: (T, B, 1, C)
        x_expanded = x.unsqueeze(2)  # (T, B, 1, C)

        # Get floor and ceil values
        # Note: We use gather for correct indexing
        idx_floor = t_floor.expand(-1, B, -1, -1)  # (T, B, Out, In)
        idx_ceil = t_ceil.expand(-1, B, -1, -1)

        # Gather (this is the key vectorized part)
        x_floor = torch.gather(x_expanded.expand(-1, -1, self.out_features, -1),
                              dim=0, index=idx_floor)
        x_ceil = torch.gather(x_expanded.expand(-1, -1, self.out_features, -1),
                             dim=0, index=idx_ceil)

        # Linear interpolation
        alpha_exp = alpha.view(1, 1, *alpha.shape)  # (1, 1, Out, In)
        interpolated = (1.0 - alpha_exp) * x_floor + alpha_exp * x_ceil

        # Apply weights and sum
        weighted = interpolated * self.weight.view(1, 1, *self.weight.shape)
        output = torch.sum(weighted, dim=-1)  # (T, B, Out)

        return output
