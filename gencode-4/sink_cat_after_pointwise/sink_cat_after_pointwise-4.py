import torch

# Define the model class
class ConcatenateModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor1, tensor2):
        # Concatenate the input tensors along the last dimension (dim=1)
        t1 = torch.cat([tensor1, tensor2], dim=1)
        
        # Reshape the concatenated tensor to have 4 channels and -1 for the other dimensions
        t2 = t1.view(t1.size(0), 4, -1)
        
        # Apply ReLU activation
        t3 = torch.relu(t2)
        
        return t3

# Initialize the model
model = ConcatenateModel()

# Generate input tensors
tensor1 = torch.randn(1, 2, 32, 32)  # A random tensor with shape (1, 2, 32, 32)
tensor2 = torch.randn(1, 2, 32, 32)  # Another random tensor with shape (1, 2, 32, 32)

# Forward pass through the model
output = model(tensor1, tensor2)

# Print output shape for verification
print(output.shape)
