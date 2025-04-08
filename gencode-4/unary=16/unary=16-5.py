import torch
import torch.nn as nn

# Define the model
class SimpleReLUModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleReLUModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation
        t2 = torch.relu(t1)  # Apply the ReLU activation function
        return t2

# Model initialization
input_size = 10  # Example input size
output_size = 5  # Example output size
model = SimpleReLUModel(input_size, output_size)

# Generate an input tensor
# For example, we can create a batch of 3 samples, each with the input size of 10
input_tensor = torch.randn(3, input_size)

# Forward pass through the model
output = model(input_tensor)

# Print output
print("Output Tensor:")
print(output)
