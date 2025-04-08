import torch

# Model definition
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(128, 64)  # Apply a linear transformation

    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = torch.relu(t1)  # Apply the ReLU activation function
        return t2

# Initializing the model
model = SimpleModel()

# Creating an input tensor
input_tensor = torch.randn(1, 128)  # Batch size of 1, 128 input features

# Forward pass through the model
output_tensor = model(input_tensor)

# Print the output
print(output_tensor)
