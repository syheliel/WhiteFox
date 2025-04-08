import torch

class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from input size 10 to output size 5
        self.other = 2.0  # Scalar value to be subtracted

    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = t1 - self.other  # Subtract 'other'
        return t2

# Initializing the model
model = LinearModel()

# Inputs to the model
input_tensor = torch.randn(1, 10)  # Batch size of 1 and input size of 10
output = model(input_tensor)

# Display the output
print(output)
