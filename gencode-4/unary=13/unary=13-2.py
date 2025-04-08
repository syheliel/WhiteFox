import torch
import torch.nn as nn
import torch.nn.functional as F

# Model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Define a linear layer with input size 10 and output size 5
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        # Apply a linear transformation to the input tensor
        t1 = self.linear(x)
        # Apply the sigmoid function to the output of the linear transformation
        t2 = torch.sigmoid(t1)
        # Multiply the output of the linear transformation by the output of the sigmoid function
        t3 = t1 * t2
        return t3

# Initializing the model
model = Model()

# Inputs to the model
# Create a random input tensor with shape (batch_size, input_features)
input_tensor = torch.randn(1, 10)  # Example input tensor for a single example
output = model(input_tensor)

# Print the output
print("Output:", output)
