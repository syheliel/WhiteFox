import torch

# Define the model
class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear layer that transforms input from size 10 to size 5

    def forward(self, x):
        t1 = self.linear(x)         # Apply linear transformation
        t2 = t1 + 3                 # Add 3 to the output
        t3 = torch.clamp_min(t2, 0) # Clamp the output to a minimum of 0
        t4 = torch.clamp_max(t3, 6) # Clamp the output to a maximum of 6
        t5 = t4 / 6                 # Divide the output by 6
        return t5

# Initialize the model
model = LinearModel()

# Generate an input tensor
input_tensor = torch.randn(1, 10)  # Input tensor of shape (1, 10)

# Forward pass through the model
output = model(input_tensor)

# Display the output
print("Output:", output)
