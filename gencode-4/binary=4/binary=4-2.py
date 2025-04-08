import torch

# Model Definition
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

# Example Input Tensor
input_tensor = torch.randn(1, 10)  # Batch size of 1 and input features of size 10
other_tensor = torch.randn(1, 5)    # The tensor to be added (must match the output size of linear layer)

# Forward Pass
output = model(input_tensor, other_tensor)

print("Output Tensor:", output)
