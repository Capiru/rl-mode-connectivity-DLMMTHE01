### Models
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from alphago.config import cfg


class SimpleModel(nn.Module):
    def __init__(self,input_space = (9,9,17),output_policy = (82),output_value = (1),mlp_size = 1024,n_layers = 8, cfg = cfg) -> None:
        super().__init__()
        self.input_l = nn.Linear(math.prod(cfg.obs_size[cfg.env_type]),mlp_size)
        self.layers = {}
        self.n_layers = n_layers
        for i in range(self.n_layers):
            self.layers[f"lin_{i}"] = nn.Linear(mlp_size,mlp_size)
        self.policy_h = nn.Linear(mlp_size,cfg.action_size[cfg.env_type])
        self.value_h = nn.Linear(mlp_size,1)

    def forward(self, x):
        if len(x.shape) > 3:
            x = torch.flatten(x,1)
        else:
            x = torch.flatten(x)
        out = F.gelu(self.input_l(x))
        res = out
        for i in range(self.n_layers):
            out = F.gelu(self.layers[f"lin_{i}"](out))
            if i % 2 == 0 and i > 0:
                out += res
                res = out
        policy = F.softmax(self.policy_h(out),dim = -1)
        value = F.tanh(self.value_h(out))
        return policy, value

class SimpleConvnet(nn.Module):
    def __init__(self,input_space = (9,9,17),output_policy = (82),output_value = (1),mlp_size = 64,n_layers = 8, cfg = cfg) -> None:
        super().__init__()
        a,b,c = cfg.obs_size[cfg.env_type]
        self.layers = {}
        self.n_layers = n_layers
        self.conv1 = nn.Conv2d(c, 192, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(192, 96, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(96, 48, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(48, 32, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.mlp_input = nn.Linear(a*b*32,mlp_size)
        self.mlp1 = nn.Linear(mlp_size,mlp_size)
        self.mlp2 = nn.Linear(mlp_size,mlp_size)
        self.policy_h = nn.Linear(mlp_size,cfg.action_size[cfg.env_type])
        self.value_h = nn.Linear(mlp_size,1)
        
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        if len(x.shape) > 3:
            x = x.permute(0,3, 1, 2)
        else:
            x = x.permute(2, 0, 1)

        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        if len(x.shape) > 3:
            x = torch.flatten(x,1)
        else:
            x = torch.flatten(x)
        out = F.gelu(self.mlp_input(x))
        out = F.gelu(self.mlp1(out))
        out = F.gelu(self.mlp2(out))


        policy = F.softmax(self.policy_h(out),dim = -1)
        value = F.tanh(self.value_h(out))
        return policy, value
    
class SimpleResnet(nn.Module):
    def __init__(self,input_space = (9,9,17),output_policy = (82),output_value = (1),mlp_size = 64,n_layers = 8, cfg = cfg) -> None:
        super().__init__()
        a,b,c = cfg.obs_size[cfg.env_type]
        self.layers = {}
        self.n_layers = n_layers
        self.conv1 = nn.Conv2d(c, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.mlp_input = nn.Linear(a*b*32,mlp_size)
        self.mlp1 = nn.Linear(mlp_size,mlp_size)
        self.mlp2 = nn.Linear(mlp_size,mlp_size)
        self.policy_h = nn.Linear(mlp_size,cfg.action_size[cfg.env_type])
        self.value_h = nn.Linear(mlp_size,1)

    def forward(self, x):
        if len(x.shape) > 3:
            x = x.permute(0,3, 1, 2)
        else:
            x = x.permute(2, 0, 1)
        res = x
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        if len(x.shape) > 3:
            x = torch.flatten(x,1)
        else:
            x = torch.flatten(x)
        out = F.gelu(self.mlp_input(x))
        out = F.gelu(self.mlp1(out))
        out = F.gelu(self.mlp2(out))


        policy = F.softmax(self.policy_h(out),dim = -1)
        value = F.tanh(self.value_h(out))
        return policy, value