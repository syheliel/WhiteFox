import torch

# Define the model
class TanhModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Linear transformation with input size 10 and output size 5
        self.linear = torch.nn.Linear(10, 5)
 
    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = torch.tanh(t1)  # Apply hyperbolic tangent function
        return t2

# Initializing the model
model = TanhModel()

# Generating inputs to the model
# Creating a random tensor with batch size 1 and input size 10
input_tensor = torch.randn(1, 10)

# Forward pass through the model
output = model(input_tensor)

# Displaying the output
print(output)
