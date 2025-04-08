import torch

class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 input features to 5 output features
        self.other = 1.5  # Scalar value to be subtracted

    def forward(self, input_tensor):
        t1 = self.linear(input_tensor)  # Apply a linear transformation
        t2 = t1 - self.other  # Subtract 'other' from the output
        return t2

# Initializing the model
model = LinearModel()

# Generating an input tensor
input_tensor = torch.randn(1, 10)  # Example input tensor with batch size 1 and 10 features
output = model(input_tensor)

# Print the output for demonstration
print("Input Tensor:", input_tensor)
print("Output Tensor:", output)
