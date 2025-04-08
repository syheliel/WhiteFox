import torch

class Model(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x, other):
        t1 = self.linear(x)  # Apply a linear transformation to the input tensor
        t2 = t1 + other      # Add another tensor to the output of the linear transformation
        return t2

# Initializing the model
input_size = 10  # Example input size
output_size = 5  # Example output size
model = Model(input_size, output_size)

# Inputs to the model
x = torch.randn(1, input_size)  # Input tensor with shape (1, input_size)
other = torch.randn(1, output_size)  # Another tensor to add with shape (1, output_size)

# Forward pass through the model
output = model(x, other)
