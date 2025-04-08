import torch

class LinearClampModel(torch.nn.Module):
    def __init__(self, input_size, output_size, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation to the input tensor
        t2 = torch.clamp_min(t1, self.min_value)  # Clamp the output of the linear transformation to a minimum value
        t3 = torch.clamp_max(t2, self.max_value)  # Clamp the output of the previous operation to a maximum value
        return t3

# Model Initialization
input_size = 10  # Example input size
output_size = 5  # Example output size
min_value = 0.0  # Minimum clamp value
max_value = 1.0  # Maximum clamp value
model = LinearClampModel(input_size, output_size, min_value, max_value)

# Inputs to the model
x_input = torch.randn(1, input_size)  # Batch size of 1 and input size of 10
output = model(x_input)

print("Input Tensor:")
print(x_input)
print("Output Tensor:")
print(output)
