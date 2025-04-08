import torch

class TanhModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a linear layer that takes input of size 10 and outputs size 5
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        t1 = self.linear(x)          # Apply a linear transformation to the input tensor
        t2 = torch.tanh(t1)         # Apply the hyperbolic tangent function
        return t2

# Initializing the model
model = TanhModel()

# Generate an input tensor of shape (1, 10) to match the input size of the linear layer
input_tensor = torch.randn(1, 10)

# Get the output of the model
output_tensor = model(input_tensor)

# Print the output
print("Output Tensor:", output_tensor)
