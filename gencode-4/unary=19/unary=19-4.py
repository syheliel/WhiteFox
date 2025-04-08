import torch

# Model
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(10, 5)  # A linear transformation from 10 input features to 5 output features

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation
        t2 = torch.sigmoid(t1)  # Apply the sigmoid function
        return t2

# Initializing the model
model = SimpleModel()

# Inputs to the model
input_tensor = torch.randn(1, 10)  # Creating a random input tensor with shape (1, 10)
output = model(input_tensor)  # Forward pass

# Display output
print("Input Tensor:\n", input_tensor)
print("Output Tensor:\n", output)
