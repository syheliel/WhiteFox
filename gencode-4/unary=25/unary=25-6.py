import torch

# Model
class Model(torch.nn.Module):
    def __init__(self, input_size, output_size, negative_slope=0.01):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.negative_slope = negative_slope

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation to the input tensor
        t2 = t1 > 0  # Create a boolean tensor
        t3 = t1 * self.negative_slope  # Multiply by negative slope
        t4 = torch.where(t2, t1, t3)  # Apply the Leaky ReLU operation
        return t4

# Initializing the model with input size 10 and output size 5
m = Model(input_size=10, output_size=5)

# Inputs to the model
x1 = torch.randn(1, 10)  # Batch size of 1, input size of 10
output = m(x1)

print("Input Tensor:")
print(x1)
print("\nOutput Tensor:")
print(output)
