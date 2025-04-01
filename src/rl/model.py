### Models
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

MODEL_SIZES = {
    "small": {"n_layers": 2, "num_neurons": 32},
    "medium": {"n_layers": 3, "num_neurons": 64},
    "large": {"n_layers": 3, "num_neurons": 128},
    "xlarge": {"n_layers": 5, "num_neurons": 256},
}


class SimpleModel(nn.Module):
    def __init__(
        self,
        num_neurons=128,
        n_layers=5,
        cfg=None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_neurons = num_neurons * 4
        self.n_layers = n_layers * 2

        self.input_l = nn.Linear(
            math.prod(cfg.obs_size[cfg.env_type]), self.num_neurons
        )
        self.layers = nn.ModuleDict()

        for i in range(self.n_layers):
            self.layers[f"lin_{i}"] = nn.Linear(self.num_neurons, self.num_neurons)
        self.policy_h = nn.Linear(self.num_neurons, cfg.action_size[cfg.env_type])
        self.value_h = nn.Linear(self.num_neurons, 1)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        if len(x.shape) > 3:
            x = torch.flatten(x, 1)
        else:
            x = torch.flatten(x)
        out = F.gelu(self.input_l(x))
        res = out
        for i in range(self.n_layers):
            out = F.gelu(self.layers[f"lin_{i}"](out))
            if i % 2 == 0 and i > 0:
                out += res
                res = out
        policy = F.softmax(self.policy_h(out), dim=-1)
        value = F.tanh(self.value_h(out))
        return policy, value


class SimpleConvnet(nn.Module):
    def __init__(
        self,
        num_neurons=32,
        n_layers=8,
        cfg=None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        a, b, c = cfg.obs_size[cfg.env_type]
        self.layers = nn.ModuleDict()
        self.n_layers = n_layers
        self.conv1 = nn.Conv2d(c, num_neurons, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_neurons, num_neurons * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            num_neurons * 2, num_neurons * 3, kernel_size=3, padding=1
        )
        self.conv4 = nn.Conv2d(
            num_neurons * 3, num_neurons * 4, kernel_size=3, padding=1
        )
        self.conv5 = nn.Conv2d(
            num_neurons * 4, num_neurons * 4, kernel_size=3, padding=1
        )
        self.mlp_input = nn.Linear(a * b * num_neurons * 4, num_neurons)
        self.mlp1 = nn.Linear(num_neurons, num_neurons)
        self.mlp2 = nn.Linear(num_neurons, num_neurons)

        # Simple Head Layers
        self.policy_h = nn.Linear(num_neurons, cfg.action_size[cfg.env_type])
        self.value_h = nn.Linear(num_neurons, 1)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        if len(x.shape) > 3:
            x = x.permute(0, 3, 1, 2)
        else:
            x = x.permute(2, 0, 1)

        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        if len(x.shape) > 3:
            x = torch.flatten(x, 1)
        else:
            x = torch.flatten(x)
        out = F.gelu(self.mlp_input(x))
        out = F.gelu(self.mlp1(out))
        out = F.gelu(self.mlp2(out))

        policy = F.softmax(self.policy_h(out), dim=-1)
        value = F.tanh(self.value_h(out))
        return policy, value


class AlphaGoZeroResnet(nn.Module):
    def __init__(
        self,
        n_layers=5,
        num_neurons=256,
        cfg=None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        a, b, c = cfg.obs_size[cfg.env_type]
        self.layers = nn.ModuleDict()
        self.n_layers = n_layers
        self.num_neurons = num_neurons

        # Input Layer
        self.conv1 = nn.Conv2d(c, self.num_neurons, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(self.num_neurons)
        # Add Residual Blocks to layers ModuleDict
        for i in range(self.n_layers):
            self.layers[f"conv_{i}_1"] = nn.Conv2d(
                self.num_neurons, self.num_neurons, kernel_size=3, padding=1
            )
            self.layers[f"bn_{i}_1"] = nn.BatchNorm2d(self.num_neurons)
            self.layers[f"dout_{i}_1"] = nn.Dropout(p=cfg.dropout)
            self.layers[f"conv_{i}_2"] = nn.Conv2d(
                self.num_neurons, self.num_neurons, kernel_size=3, padding=1
            )
            self.layers[f"bn_{i}_2"] = nn.BatchNorm2d(self.num_neurons)
            self.layers[f"dout_{i}_2"] = nn.Dropout(p=cfg.dropout)

        # Policy Head
        # TODO: why just limiting to two neurons here? is this limiting the model capacity?
        self.policy_h_1 = nn.Conv2d(self.num_neurons, 2, kernel_size=1, padding=0)
        self.policy_bn_1 = nn.BatchNorm2d(2)
        self.policy_h = nn.Linear(2 * a * b, cfg.action_size[cfg.env_type])

        # Value Head
        # TODO: same comment as above
        self.value_h_1 = nn.Conv2d(self.num_neurons, 1, kernel_size=1, padding=0)
        self.value_bn_1 = nn.BatchNorm2d(1)
        self.value_mlp_1 = nn.Linear(a * b, self.num_neurons)
        self.value_h = nn.Linear(self.num_neurons, 1)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        if len(x.shape) > 3:
            x = x.permute(0, 3, 1, 2)
        else:
            x = x.permute(2, 0, 1)
            x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
        x = F.leaky_relu(self.batch_norm1(self.conv1(x)))
        res = x
        for i in range(self.n_layers):
            x = F.leaky_relu(self.layers[f"bn_{i}_1"](self.layers[f"conv_{i}_1"](x)))
            x = self.layers[f"dout_{i}_1"](x)
            x = self.layers[f"bn_{i}_2"](self.layers[f"conv_{i}_2"](x))
            x = x + res
            res = x
            x = F.leaky_relu(x)
            x = self.layers[f"dout_{i}_2"](x)

        p_out = F.leaky_relu(self.policy_bn_1(self.policy_h_1(x)))
        v_out = F.leaky_relu(self.value_bn_1(self.value_h_1(x)))

        v_mlp = F.leaky_relu(self.value_mlp_1(torch.flatten(v_out, 1)))

        policy = F.softmax(self.policy_h(torch.flatten(p_out, 1)), dim=-1)
        value = F.tanh(self.value_h(torch.flatten(v_mlp, 1)))
        if policy.shape[0] == 1:
            policy = policy.reshape(policy.shape[1])
            value = value.reshape(value.shape[1])

        return policy, value


class SimpleResnet(nn.Module):
    def __init__(
        self,
        mlp_size=64,
        n_layers=8,
        cfg=None,
    ) -> None:
        super().__init__()
        a, b, c = cfg.obs_size[cfg.env_type]
        self.layers = nn.ModuleDict()
        self.n_layers = n_layers
        self.conv1 = nn.Conv2d(c, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.mlp_input = nn.Linear(a * b * 32, mlp_size)
        self.mlp1 = nn.Linear(mlp_size, mlp_size)
        self.mlp2 = nn.Linear(mlp_size, mlp_size)
        self.policy_h = nn.Linear(mlp_size, cfg.action_size[cfg.env_type])
        self.value_h = nn.Linear(mlp_size, 1)

    def forward(self, x):
        if len(x.shape) > 3:
            x = x.permute(0, 3, 1, 2)
        else:
            x = x.permute(2, 0, 1)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        if len(x.shape) > 3:
            x = torch.flatten(x, 1)
        else:
            x = torch.flatten(x)
        out = F.gelu(self.mlp_input(x))
        out = F.gelu(self.mlp1(out))
        out = F.gelu(self.mlp2(out))

        policy = F.softmax(self.policy_h(out), dim=-1)
        value = F.tanh(self.value_h(out))
        return policy, value
