import torch

# Model definition
class LeakyReLUModel(torch.nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 input features to 5 output features
        self.negative_slope = negative_slope

    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = t1 > 0  # Create boolean tensor for positive values
        t3 = t1 * self.negative_slope  # Multiply by negative slope for negative values
        t4 = torch.where(t2, t1, t3)  # Apply Leaky ReLU logic
        return t4

# Initializing the model
model = LeakyReLUModel()

# Inputs to the model
input_tensor = torch.randn(1, 10)  # Batch size of 1 and 10 input features
output_tensor = model(input_tensor)

print("Input Tensor:")
print(input_tensor)
print("Output Tensor:")
print(output_tensor)
