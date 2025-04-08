import torch

# Model Definition
class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear layer from 10 input features to 5 output features

    def forward(self, x, other):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = t1 + other      # Add another tensor to the output of the linear transformation
        return t2

# Initializing the model
model = LinearModel()

# Generating input tensor
input_tensor = torch.randn(1, 10)  # Batch size of 1 and 10 features
other_tensor = torch.randn(1, 5)    # Tensor to be added, matching the output size of the linear layer

# Forward pass
output = model(input_tensor, other_tensor)

# Display the output
print(output)
