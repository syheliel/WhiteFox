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
        t2 = torch.clamp_min(t1, self.min_value)  # Clamp the output to a minimum value
        t3 = torch.clamp_max(t2, self.max_value)  # Clamp the output to a maximum value
        return t3

# Initialize the model with specific dimensions and clamp values
input_dim = 10  # Example input dimension
output_dim = 5  # Example output dimension
min_value = 0.0  # Minimum value for clamping
max_value = 1.0  # Maximum value for clamping
model = ClampingModel(input_dim, output_dim, min_value, max_value)

# Create an input tensor
input_tensor = torch.randn(1, input_dim)  # Batch size of 1 and input dimension of 10

# Forward pass through the model
output_tensor = model(input_tensor)

# Print the output tensor
print(output_tensor)
