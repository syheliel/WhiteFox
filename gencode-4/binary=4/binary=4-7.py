import torch

class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear layer with input size 10 and output size 5

    def forward(self, x, other):
        t1 = self.linear(x)      # Apply linear transformation
        t2 = t1 + other          # Add another tensor to the output
        return t2

# Initializing the model
model = LinearModel()

# Inputs to the model
input_tensor = torch.randn(1, 10)  # Batch size of 1, input size of 10
other_tensor = torch.randn(1, 5)    # Batch size of 1, size matching the output of the linear layer

# Get the output of the model
output = model(input_tensor, other_tensor)

print(output)
