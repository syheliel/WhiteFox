import torch

# Define the model
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Define a linear layer with input features 128 and output features 64
        self.linear = torch.nn.Linear(128, 64)

    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = t1.permute(0, 2, 1)  # Permute the output tensor (assuming 3D input)
        return t2

# Initializing the model
m = Model()

# Create an input tensor with shape (batch_size, features), e.g., (1, 128)
x1 = torch.randn(1, 128)

# Get the output from the model
output = m(x1)

print(output)
