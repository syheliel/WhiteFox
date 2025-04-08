import torch

# Model definition
class LinearReLUModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 to 5 dimensions
        self.other_tensor = torch.randn(1, 5)  # A tensor to be added to the output of the linear layer

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation to the input tensor
        t2 = t1 + self.other_tensor  # Add another tensor to the output of the linear transformation
        t3 = torch.relu(t2)  # Apply the ReLU activation function to the result
        return t3

# Initialize the model
model = LinearReLUModel()

# Input tensor
input_tensor = torch.randn(1, 10)  # Batch size of 1 and 10 features

# Forward pass through the model
output = model(input_tensor)

# Display the output
print(output)
