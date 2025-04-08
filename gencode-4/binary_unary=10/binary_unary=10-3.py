import torch

class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a linear layer with input size 10 and output size 5
        self.linear = torch.nn.Linear(10, 5)
        # Define another tensor for addition, initialized with ones
        self.other = torch.ones(1, 5)  # Shape must match the output of the linear layer

    def forward(self, x):
        t1 = self.linear(x)        # Apply a linear transformation
        t2 = t1 + self.other       # Add another tensor to the output
        t3 = torch.relu(t2)        # Apply the ReLU activation function
        return t3

# Initializing the model
model = LinearModel()

# Input tensor to the model
input_tensor = torch.randn(1, 10)  # Batch size of 1 and input features of size 10
output = model(input_tensor)

# Print the output
print(output)
