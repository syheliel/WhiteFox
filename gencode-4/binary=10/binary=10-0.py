import torch

class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 to 5 dimensions

    def forward(self, x1, other):
        t1 = self.linear(x1)  # Apply a linear transformation
        t2 = t1 + other  # Add another tensor to the output of the linear transformation
        return t2

# Initializing the model
model = LinearModel()

# Input tensor for the model
x1 = torch.randn(1, 10)  # Batch size of 1 and input features of size 10
other = torch.ones(1, 5)  # Another tensor to add, with the same size as the output of the linear layer

# Generating the output
output = model(x1, other)

print("Output:", output)
