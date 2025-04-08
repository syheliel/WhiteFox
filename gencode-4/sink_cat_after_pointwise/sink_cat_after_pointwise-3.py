import torch

# Define the model
class ConcatenationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 20)  # First linear layer
        self.fc2 = torch.nn.Linear(10, 20)  # Second linear layer

    def forward(self, tensor1, tensor2):
        # Concatenate the two input tensors along dimension 1
        t1 = torch.cat([tensor1, tensor2], dim=1)
        # Reshape the concatenated tensor
        t2 = t1.view(-1, 20)  # Assuming we want to reshape to have 20 features
        # Apply ReLU activation function
        t3 = torch.relu(t2)
        return t3

# Initializing the model
model = ConcatenationModel()

# Inputs to the model
tensor1 = torch.randn(2, 10)  # Input tensor 1 of shape (2, 10)
tensor2 = torch.randn(2, 10)  # Input tensor 2 of shape (2, 10)

# Forward pass
output = model(tensor1, tensor2)

# Display the output
print(output)
