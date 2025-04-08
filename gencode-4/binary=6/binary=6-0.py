import torch

class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a linear layer that transforms input from 10 to 5 dimensions
        self.linear = torch.nn.Linear(10, 5)
        # Define a constant tensor to subtract from the output
        self.other = torch.tensor([0.5, 1.0, -0.5, 0.0, 0.25], dtype=torch.float32)

    def forward(self, x):
        # Apply the linear transformation
        t1 = self.linear(x)
        # Subtract 'other' from the output of the linear transformation
        t2 = t1 - self.other
        return t2

# Initializing the model
model = LinearModel()

# Inputs to the model (batch size of 1, 10 features)
input_tensor = torch.randn(1, 10)

# Forward pass through the model
output = model(input_tensor)

# Output
print("Input Tensor:", input_tensor)
print("Output Tensor:", output)
