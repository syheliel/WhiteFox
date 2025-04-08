import torch

class LinearModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        # Initialize another tensor to add after the linear transformation
        self.other = torch.randn(1, output_size)  # Random tensor for addition

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation
        t2 = t1 + self.other  # Add another tensor to the output
        return t2

# Initializing the model
input_size = 10  # Size of the input features
output_size = 5  # Size of the output features
model = LinearModel(input_size, output_size)

# Input tensor to the model
x_input = torch.randn(1, input_size)  # Batch size of 1 and input size of 10
output = model(x_input)

print("Output tensor:", output)
