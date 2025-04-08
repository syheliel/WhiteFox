import torch

# Define the model
class SplitSqueezeModel(torch.nn.Module):
    def __init__(self, split_dim, num_chunks):
        super().__init__()
        self.split_dim = split_dim
        self.split_sizes = [1] * num_chunks  # Ensure all sizes are 1

    def forward(self, input_tensor):
        # Split the input tensor into chunks along the specified dimension
        split_tensor = torch.split(input_tensor, self.split_sizes, dim=self.split_dim)
        # Squeeze each chunk of the split tensor along the same dimension
        squeezed_tensors = [torch.squeeze(t, dim=self.split_dim) for t in split_tensor]
        return squeezed_tensors

# Initialize the model
split_dim = 1  # Dimension to split along
num_chunks = 5  # Number of chunks to split into
model = SplitSqueezeModel(split_dim, num_chunks)

# Create an input tensor
# Example: Batch size 1, 5 channels, 64 height, 64 width
input_tensor = torch.randn(1, 5, 64, 64)  # Make sure to have enough channels for splitting

# Apply the model to the input tensor
output = model(input_tensor)

# Print the output shapes to verify
for i, tensor in enumerate(output):
    print(f'Squeezed tensor {i} shape: {tensor.shape}')
