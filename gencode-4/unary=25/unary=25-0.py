import torch

# Model Definition
class LeakyReLUModel(torch.nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear layer with input size 10 and output size 5
        self.negative_slope = negative_slope

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation to the input tensor
        t2 = t1 > 0  # Create a boolean tensor where each element is True if t1 > 0
        t3 = t1 * self.negative_slope  # Multiply the output of the linear transformation by the negative slope
        t4 = torch.where(t2, t1, t3)  # Apply Leaky ReLU logic
        return t4

# Initializing the model
model = LeakyReLUModel(negative_slope=0.01)

# Inputs to the model
input_tensor = torch.randn(1, 10)  # Batch size of 1 and input dimension of 10
output_tensor = model(input_tensor)

# Output the model output
print("Output Tensor:\n", output_tensor)
