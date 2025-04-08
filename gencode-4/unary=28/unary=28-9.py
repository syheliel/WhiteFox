import torch

# Define the model
class ClampingModel(torch.nn.Module):
    def __init__(self, in_features, out_features, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.min_value = min_value
        self.max_value = max_value
    
    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation
        t2 = torch.clamp_min(t1, self.min_value)  # Clamp to minimum value
        t3 = torch.clamp_max(t2, self.max_value)  # Clamp to maximum value
        return t3

# Initialize the model with example parameters
in_features = 10  # Input features
out_features = 5  # Output features
min_value = 0.0   # Minimum clamp value
max_value = 1.0   # Maximum clamp value
model = ClampingModel(in_features, out_features, min_value, max_value)

# Generate an input tensor
input_tensor = torch.randn(1, in_features)  # Batch size of 1

# Get the output from the model
output = model(input_tensor)

# Print the output tensor
print(output)
