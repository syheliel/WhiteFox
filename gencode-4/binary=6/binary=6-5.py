import torch

# Model definition
class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 to 5 features
        self.other = torch.tensor(1.5)  # Scalar value to subtract from the output

    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = t1 - self.other  # Subtract the scalar 'other'
        return t2

# Initializing the model
model = CustomModel()

# Generating an input tensor
input_tensor = torch.randn(1, 10)  # A batch of 1 with 10 features
output = model(input_tensor)

# Print the output
print(output)
