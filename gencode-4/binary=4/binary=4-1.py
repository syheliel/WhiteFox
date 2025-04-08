import torch

# Model
class LinearAddModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 to 5 dimensions

    def forward(self, input_tensor, other):
        t1 = self.linear(input_tensor)  # Apply linear transformation
        t2 = t1 + other  # Add another tensor
        return t2

# Initializing the model
model = LinearAddModel()

# Inputs to the model
input_tensor = torch.randn(1, 10)  # Batch size of 1, input dimension of 10
other = torch.randn(1, 5)  # The tensor to be added, must match the output dimension of the linear layer

# Forward pass
output = model(input_tensor, other)

# Output the result
print(output)
