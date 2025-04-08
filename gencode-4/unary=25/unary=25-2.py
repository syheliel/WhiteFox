import torch

class LeakyReLUModel(torch.nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 input features to 5 output features
        self.negative_slope = negative_slope  # Negative slope for the Leaky ReLU

    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = t1 > 0  # Create a boolean tensor
        t3 = t1 * self.negative_slope  # Multiply by negative slope
        t4 = torch.where(t2, t1, t3)  # Apply the condition
        return t4

# Initializing the model
model = LeakyReLUModel(negative_slope=0.01)

# Generate an input tensor
input_tensor = torch.randn(1, 10)  # A random input tensor with shape (1, 10)

# Get the model output
output_tensor = model(input_tensor)

print("Input Tensor:")
print(input_tensor)
print("Output Tensor:")
print(output_tensor)
