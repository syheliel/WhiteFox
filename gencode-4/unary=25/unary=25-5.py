import torch

# Model definition
class LeakyReLUModel(torch.nn.Module):
    def __init__(self, input_size, output_size, negative_slope=0.01):
        super(LeakyReLUModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.negative_slope = negative_slope

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation to the input tensor
        t2 = t1 > 0  # Create a boolean tensor where each element is True if t1 > 0
        t3 = t1 * self.negative_slope  # Multiply the output of the linear transformation by the negative slope
        t4 = torch.where(t2, t1, t3)  # Apply Leaky ReLU
        return t4

# Initializing the model
input_size = 10  # Example input feature size
output_size = 5  # Example output feature size
model = LeakyReLUModel(input_size, output_size)

# Generate an input tensor
x_input = torch.randn(1, input_size)  # Batch size of 1 with input feature size of 10

# Forward pass through the model
output = model(x_input)

print("Input Tensor:")
print(x_input)
print("Output Tensor:")
print(output)
