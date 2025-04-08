import torch

# Define the model
class GatedLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Linear transformation from input size 10 to output size 5
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation
        t2 = torch.sigmoid(t1)  # Apply the sigmoid function
        t3 = t1 * t2  # Multiply the linear output by the sigmoid output
        return t3

# Initializing the model
model = GatedLinearModel()

# Generate input tensor
input_tensor = torch.randn(1, 10)  # Batch size of 1, input size of 10

# Get the output of the model
output = model(input_tensor)

# Print the output
print(output)
