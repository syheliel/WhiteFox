import torch

# Model definition
class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    
    def forward(self, x, other):
        t1 = self.linear(x)  # Apply a linear transformation
        t2 = t1 + other      # Add another tensor to the output
        return t2

# Initializing the model
input_dim = 10  # Example input dimension
output_dim = 5  # Example output dimension
model = Model(input_dim, output_dim)

# Creating an input tensor and another tensor to add
x = torch.randn(1, input_dim)  # Input tensor
other = torch.randn(1, output_dim)  # Another tensor to add

# Forward pass
output = model(x, other)

# Printing the output
print(output)
