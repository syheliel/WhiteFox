import torch

class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 features to 5 features
    
    def forward(self, x, other):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = t1 + other  # Add another tensor to the output of the linear transformation
        return t2

# Initializing the model
model = LinearModel()

# Inputs to the model
x_input = torch.randn(1, 10)  # A random input tensor with shape (1, 10)
other_tensor = torch.randn(1, 5)  # Another tensor to add, with shape (1, 5)

# Forward pass through the model
output = model(x_input, other_tensor)
