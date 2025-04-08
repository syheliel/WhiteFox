import torch

# Model
class ClampModel(torch.nn.Module):
    def __init__(self, in_features, out_features, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation to the input tensor
        t2 = torch.clamp_min(t1, self.min_value)  # Clamp the output of the linear transformation to a minimum value
        t3 = torch.clamp_max(t2, self.max_value)  # Clamp the output of the previous operation to a maximum value
        return t3

# Parameters
in_features = 10  # Number of input features
out_features = 5  # Number of output features
min_value = 0.0   # Minimum value for clamping
max_value = 1.0   # Maximum value for clamping

# Initializing the model
model = ClampModel(in_features, out_features, min_value, max_value)

# Inputs to the model
input_tensor = torch.randn(1, in_features)  # Batch size of 1
output_tensor = model(input_tensor)

# Print the output tensor
print("Output Tensor:", output_tensor)
