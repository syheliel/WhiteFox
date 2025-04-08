import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x, other):
        t1 = self.linear(x)      # Apply a linear transformation to the input tensor
        t2 = t1 + other          # Add another tensor to the output of the linear transformation
        t3 = torch.relu(t2)      # Apply the ReLU activation function to the result
        return t3

# Initialize the model
input_dim = 10  # Example input dimension
output_dim = 5  # Example output dimension
model = Model(input_dim, output_dim)

# Create an input tensor and the 'other' tensor
x = torch.randn(1, input_dim)          # Input tensor with shape (1, input_dim)
other = torch.randn(1, output_dim)     # Other tensor with shape (1, output_dim)

# Get the output of the model
output = model(x, other)

# Display the output
print(output)
