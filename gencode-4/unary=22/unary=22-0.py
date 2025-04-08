import torch

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 input features to 5 output features

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation
        t2 = torch.tanh(t1)  # Apply the hyperbolic tangent function
        return t2

# Initialize the model
model = SimpleModel()

# Generate an input tensor
input_tensor = torch.randn(1, 10)  # A batch size of 1 with 10 input features

# Pass the input through the model
output = model(input_tensor)

# Print the output
print(output)
