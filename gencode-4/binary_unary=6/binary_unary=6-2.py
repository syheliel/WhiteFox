import torch
import torch.nn.functional as F

class LinearReLUModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a linear layer with input size 10 and output size 5
        self.linear = torch.nn.Linear(10, 5)
        self.other = torch.tensor(0.5)  # Define the constant 'other' to be subtracted

    def forward(self, x):
        t1 = self.linear(x)            # Apply linear transformation
        t2 = t1 - self.other           # Subtract 'other'
        t3 = F.relu(t2)                # Apply ReLU activation function
        return t3

# Initializing the model
model = LinearReLUModel()

# Generate input tensor
# Input tensor of shape (batch_size, input_features)
input_tensor = torch.randn(1, 10)  # For example, a batch size of 1 with 10 features

# Get the output of the model
output = model(input_tensor)

# Print the output
print(output)
