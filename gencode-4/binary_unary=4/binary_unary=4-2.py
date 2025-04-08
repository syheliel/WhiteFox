import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, other):
        t1 = self.linear(x)  # Apply a linear transformation to the input tensor
        t2 = t1 + other      # Add another tensor to the output of the linear transformation
        t3 = F.relu(t2)      # Apply the ReLU activation function to the result
        return t3

# Initializing the model
input_dim = 10  # Example input dimension
output_dim = 5  # Example output dimension
model = Model(input_dim, output_dim)

# Creating the input tensor
x = torch.randn(1, input_dim)  # Batch size of 1
other = torch.randn(1, output_dim)  # Another tensor to be added

# Forward pass
output = model(x, other)

print("Output Tensor:")
print(output)
