import torch

class ClampingModel(torch.nn.Module):
    def __init__(self, in_features, out_features, min_value=0.0, max_value=1.0):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation
        t2 = torch.clamp_min(t1, self.min_value)  # Clamp to minimum value
        t3 = torch.clamp_max(t2, self.max_value)  # Clamp to maximum value
        return t3

# Initialize the model
model = ClampingModel(in_features=10, out_features=5, min_value=0.0, max_value=1.0)

# Generate an input tensor
input_tensor = torch.randn(1, 10)  # Batch size of 1 and 10 features

# Forward pass
output = model(input_tensor)
print(output)
