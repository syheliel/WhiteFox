import torch

# Model definition
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 to 5 dimensions

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation
        t2 = torch.tanh(t1)  # Apply the hyperbolic tangent function
        return t2

# Initializing the model
model = SimpleModel()

# Generating an input tensor
input_tensor = torch.randn(1, 10)  # Batch size of 1 and input dimension of 10

# Passing the input tensor through the model
output = model(input_tensor)

# Print the output
print(output)
