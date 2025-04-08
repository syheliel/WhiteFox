import torch

# Model Definition
class LeakyReLUModel(torch.nn.Module):
    def __init__(self, input_size, output_size, negative_slope=0.01):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.negative_slope = negative_slope

    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = t1 > 0  # Create boolean tensor
        t3 = t1 * self.negative_slope  # Multiply by negative slope
        t4 = torch.where(t2, t1, t3)  # Apply Leaky ReLU logic
        return t4

# Initializing the model
input_size = 10  # Example input size
output_size = 5  # Example output size
model = LeakyReLUModel(input_size, output_size)

# Inputs to the model
input_tensor = torch.randn(1, input_size)  # Example input tensor with batch size of 1
output_tensor = model(input_tensor)  # Forward pass

# Print the output
print("Input Tensor:")
print(input_tensor)
print("Output Tensor:")
print(output_tensor)
