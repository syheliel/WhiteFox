import torch

# Define the model
class ConcatenateAndReluModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor1, tensor2):
        # Concatenate tensors along the channel dimension (dim=1)
        t1 = torch.cat([tensor1, tensor2], dim=1)
        # Reshape the concatenated tensor to (batch_size, -1) to flatten the spatial dimensions
        t2 = t1.view(t1.size(0), -1)
        # Apply ReLU activation function
        t3 = torch.relu(t2)
        return t3

# Initializing the model
model = ConcatenateAndReluModel()

# Generating input tensors
input_tensor1 = torch.randn(1, 3, 64, 64)  # Example input tensor with shape (batch_size, channels, height, width)
input_tensor2 = torch.randn(1, 3, 64, 64)  # Another input tensor with the same shape

# Forward pass
output = model(input_tensor1, input_tensor2)

# Output shape
print(output.shape)  # Should print the shape of the output tensor
