import torch

# Model
class LeakyReLUModel(torch.nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from input size 10 to output size 5
        self.negative_slope = negative_slope

    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = t1 > 0  # Create boolean tensor
        t3 = t1 * self.negative_slope  # Multiply by negative slope
        t4 = torch.where(t2, t1, t3)  # Apply Leaky ReLU logic
        return t4

# Initializing the model
model = LeakyReLUModel(negative_slope=0.01)

# Inputs to the model
input_tensor = torch.randn(1, 10)  # Batch size of 1, input size of 10
output = model(input_tensor)

print("Input Tensor:")
print(input_tensor)
print("Output Tensor:")
print(output)
