import torch

class ClampingModel(torch.nn.Module):
    def __init__(self, input_size, output_size, min_value=0.0, max_value=1.0):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation
        t2 = torch.clamp_min(t1, self.min_value)  # Clamp to minimum value
        t3 = torch.clamp_max(t2, self.max_value)  # Clamp to maximum value
        return t3

# Initializing the model
input_size = 10  # For example, input feature size
output_size = 5  # For example, output feature size
model = ClampingModel(input_size, output_size, min_value=0.0, max_value=1.0)

# Generating the input tensor
x_input = torch.randn(1, input_size)  # Batch size of 1 and input size of 10
output = model(x_input)

# Output the result
print("Input Tensor:", x_input)
print("Output Tensor:", output)
