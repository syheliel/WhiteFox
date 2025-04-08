import torch

class ClampingModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation
        t2 = torch.clamp_min(t1, self.min_value)  # Clamp to minimum value
        t3 = torch.clamp_max(t2, self.max_value)  # Clamp to maximum value
        return t3

# Initialize the model
input_dim = 10  # Number of input features
output_dim = 5  # Number of output features
min_value = 0.0  # Minimum value for clamping
max_value = 1.0  # Maximum value for clamping
model = ClampingModel(input_dim, output_dim, min_value, max_value)

# Generate input tensor
x1 = torch.randn(1, input_dim)  # Batch size of 1
output = model(x1)

print("Input Tensor:", x1)
print("Output Tensor:", output)
