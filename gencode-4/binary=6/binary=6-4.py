import torch

# Define the model
class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a linear layer with 10 input features and 5 output features
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        # Apply linear transformation
        t1 = self.linear(x)
        # Define the scalar 'other' to subtract
        other = torch.tensor(0.5)  # Subtracting 0.5 from the output
        # Subtract 'other' from the output of the linear transformation
        t2 = t1 - other
        return t2

# Initializing the model
model = LinearModel()

# Inputs to the model: Create a random tensor with the shape (batch_size, input_features)
input_tensor = torch.randn(1, 10)  # Batch size of 1, and 10 input features

# Get the output of the model
output = model(input_tensor)

print("Input Tensor:")
print(input_tensor)
print("\nOutput Tensor:")
print(output)
