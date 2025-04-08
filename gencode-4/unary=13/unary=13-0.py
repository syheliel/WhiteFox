import torch
import torch.nn as nn

# Model definition
class GatingModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GatingModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        t1 = self.linear(x)          # Apply a linear transformation
        t2 = torch.sigmoid(t1)      # Apply the sigmoid function
        t3 = t1 * t2                 # Multiply the linear output by the sigmoid output
        return t3

# Initializing the model
input_dim = 10   # Example input feature dimension
output_dim = 5   # Example output feature dimension
model = GatingModel(input_dim, output_dim)

# Inputs to the model
x_input = torch.randn(1, input_dim)  # Batch size of 1, with input_dim features
output = model(x_input)

print("Input tensor:", x_input)
print("Output tensor:", output)
