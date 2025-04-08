import torch
import torch.nn as nn
import torch.nn.functional as F

# Model
class GatingModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(GatingModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
 
    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation to the input tensor
        t2 = torch.sigmoid(t1)  # Apply the sigmoid function to the output of the linear transformation
        t3 = t1 * t2  # Multiply the output of the linear transformation by the output of the sigmoid function
        return t3

# Initializing the model with an example input size of 10 and output size of 5
input_size = 10
output_size = 5
model = GatingModel(input_size, output_size)

# Generating input tensor for the model
input_tensor = torch.randn(1, input_size)  # Batch size of 1 and input size of 10
output_tensor = model(input_tensor)

# Displaying the output
print("Output Tensor:", output_tensor)
