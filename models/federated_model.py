import torch
import torch.nn as nn
import copy


class FederatedModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.model_version = 0

    def forward(self, x):
        return self.base_model(x)

    def get_parameters(self):
        return copy.deepcopy(self.state_dict())

    def set_parameters(self, state_dict):
        self.load_state_dict(state_dict)

    def get_parameter_vector(self):
        params = []
        for param in self.parameters():
            params.append(param.data.flatten())
        return torch.cat(params)

    def set_parameter_vector(self, param_vector):
        offset = 0
        for param in self.parameters():
            param_size = param.data.numel()
            param.data = param_vector[offset:offset + param_size].reshape(param.data.shape)
            offset += param_size

    def compute_gradient_norm(self):
        total_norm = 0.0
        for param in self.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def zero_gradients(self):
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()

    def clone(self):
        return copy.deepcopy(self)

    def increment_version(self):
        self.model_version += 1

    def get_version(self):
        return self.model_version

