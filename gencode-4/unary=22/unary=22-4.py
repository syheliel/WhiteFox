import torch

# Define the model
class TanhModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a linear layer with input size 10 and output size 5
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation to the input tensor
        t2 = torch.tanh(t1)  # Apply the hyperbolic tangent function to the output of the linear transformation
        return t2

# Initializing the model
model = TanhModel()

# Input to the model: a tensor with shape (1, 10)
input_tensor = torch.randn(1, 10)

# Forward pass through the model
output = model(input_tensor)

# To verify the output
print("Input Tensor:", input_tensor)
print("Output Tensor:", output)
