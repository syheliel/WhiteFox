import torch
import torch.nn as nn
import torch.nn.functional as F

# Model
class GatingModel(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.linear = nn.Linear(input_features, output_features)

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation to the input tensor
        t2 = torch.sigmoid(t1)  # Apply the sigmoid function
        t3 = t1 * t2  # Multiply the output of the linear transformation by the output of the sigmoid function
        return t3

# Initializing the model
input_features = 10  # Number of input features
output_features = 5  # Number of output features
model = GatingModel(input_features, output_features)

# Inputs to the model
x_input = torch.randn(1, input_features)  # Batch size of 1 and input features
output = model(x_input)

print("Input Tensor:", x_input)
print("Output Tensor:", output)
