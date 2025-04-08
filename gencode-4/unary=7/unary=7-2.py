import torch

class ReLU6Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a linear transformation layer
        self.linear = torch.nn.Linear(10, 10)  # Input and output sizes are both 10 for simplicity

    def forward(self, x):
        l1 = self.linear(x)  # Apply the linear transformation
        l2 = l1 + 3          # Add 3 to the output of the linear transformation
        l3 = l2.clamp(min=0) # Clamp the output to a minimum of 0
        l4 = l3.clamp(max=6) # Clamp the output to a maximum of 6
        l5 = l1 * l4         # Multiply the output of the linear transformation by the clamped output
        l6 = l5 / 6          # Divide the result by 6
        return l6

# Initializing the model
model = ReLU6Model()

# Generate an input tensor
input_tensor = torch.randn(1, 10)  # Batch size of 1, input feature size of 10

# Get the output from the model
output = model(input_tensor)

# Display the output
print(output)
