import torch

class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 to 5 dimensions

    def forward(self, x, other):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = t1 + other      # Add another tensor to the output
        return t2

# Initializing the model
model = LinearModel()

# Input tensor
input_tensor = torch.randn(1, 10)  # Batch size of 1 and 10 features
other_tensor = torch.randn(1, 5)    # The tensor to add (must match the output size of the linear layer)

# Getting the output from the model
output = model(input_tensor, other_tensor)

print(output)
