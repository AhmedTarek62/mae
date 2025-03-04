import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

class PrefixTuning(nn.Module):
    def __init__(self, prefix_length, embed_dim, num_layers, device):
        super(PrefixTuning, self).__init__()
        self.prefix_length = prefix_length
        self.num_layers = num_layers
        self.prefixes = nn.ParameterList([
            nn.Parameter(torch.randn(1, prefix_length, embed_dim))
            for _ in range(num_layers)
        ]).to(device)


    def forward(self, x):
        batch_size = x.size(0)
        print(f"batch_size: {batch_size}")
        print(f"num_layers: {self.num_layers}")
        for i in range(self.num_layers):
            print(f"i: {i}")
            prefix = self.prefixes[i].expand(batch_size, -1, -1)
            print(f"prefix: {prefix.shape}")
            print(f"x: {x.shape}")
            x = torch.cat((prefix, x), dim=1)
        return x


def create_prefix_tuned_vit(model: VisionTransformer, prefix_length=10, device='cpu'):
    embed_dim = model.embed_dim
    num_layers = len(model.blocks)
    prefix_tuning = PrefixTuning(prefix_length, embed_dim, num_layers, device)

    # Freeze all params
    for param in model.blocks.parameters():
        param.requires_grad = False

    # Unfreeze classifier layer
    for param in model.head.parameters():
        param.requires_grad = True

    # Integrate prefix tuning into the model
    original_forward = model.forward

    def forward_with_prefix(x):
        print(f"x.shape: {x.shape}")
        x = prefix_tuning(x)
        return original_forward(x)

    model.forward = forward_with_prefix
    return model