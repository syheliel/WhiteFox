import torch

# Define the model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 to 5 dimensions
        self.other = torch.randn(1, 5)         # Another tensor to be added (1, 5) shape

    def forward(self, x):
        t1 = self.linear(x)            # Apply linear transformation
        t2 = t1 + self.other           # Add another tensor
        t3 = torch.relu(t2)            # Apply ReLU activation
        return t3

# Initializing the model
model = Model()

# Generate input tensor
input_tensor = torch.randn(1, 10)  # Input tensor of shape (1, 10)

# Get the output from the model
output = model(input_tensor)

# Print the output
print(output)
