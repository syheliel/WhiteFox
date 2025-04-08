import torch

# Model definition
class ReLU6Model(torch.nn.Module):
    def __init__(self):
        super(ReLU6Model, self).__init__()
        self.linear = torch.nn.Linear(10, 5)  # Pointwise linear transformation

    def forward(self, x):
        l1 = self.linear(x)  # Apply linear transformation
        l2 = l1 + 3  # Add 3 to the output of the linear transformation
        l3 = torch.clamp(l2, min=0)  # Clamp the output to a minimum of 0
        l4 = torch.clamp(l3, max=6)  # Clamp the output to a maximum of 6
        l5 = l1 * l4  # Multiply the output of the linear transformation by the clamped output
        l6 = l5 / 6  # Divide the output by 6
        return l6

# Initializing the model
model = ReLU6Model()

# Inputs to the model
input_tensor = torch.randn(1, 10)  # Example input tensor of shape (1, 10)
output_tensor = model(input_tensor)

# Displaying the output
print("Input Tensor:\n", input_tensor)
print("Output Tensor:\n", output_tensor)
