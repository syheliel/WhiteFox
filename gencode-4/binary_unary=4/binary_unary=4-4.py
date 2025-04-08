import torch

# Model definition
class CustomModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x, other):
        t1 = self.linear(x)       # Apply a linear transformation to the input tensor
        t2 = t1 + other           # Add another tensor to the output of the linear transformation
        t3 = torch.relu(t2)       # Apply the ReLU activation function to the result
        return t3

# Initializing the model with input and output dimensions
input_dim = 10  # Example input dimension
output_dim = 5  # Example output dimension
model = CustomModel(input_dim, output_dim)

# Inputs to the model
x = torch.randn(1, input_dim)       # Input tensor with batch size 1
other = torch.randn(1, output_dim)   # Other tensor to add

# Forward pass
output = model(x, other)

# Print the output
print(output)
