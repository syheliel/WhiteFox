import torch

# Define the model
class ClampingModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation to the input tensor
        t2 = torch.clamp_min(t1, self.min_value)  # Clamp the output of the linear transformation to a minimum value
        t3 = torch.clamp_max(t2, self.max_value)  # Clamp the output of the previous operation to a maximum value
        return t3

# Model hyperparameters
input_dim = 10
output_dim = 5
min_value = 0.0
max_value = 1.0

# Initializing the model
model = ClampingModel(input_dim, output_dim, min_value, max_value)

# Generate an input tensor
input_tensor = torch.randn(1, input_dim)  # Batch size of 1 and input dimension of 10

# Getting the output from the model
output = model(input_tensor)

print("Input Tensor:", input_tensor)
print("Output Tensor:", output)
