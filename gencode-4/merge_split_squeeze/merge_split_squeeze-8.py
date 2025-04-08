import torch

class SplitSqueezeModel(torch.nn.Module):
    def __init__(self, split_dim):
        super().__init__()
        self.split_dim = split_dim  # Dimension to split along

    def forward(self, x):
        # Split the input tensor into chunks of size 1 along the specified dimension
        split_tensor = torch.split(x, 1, dim=self.split_dim)
        # Squeeze each chunk of the split tensor along the same dimension
        squeezed_tensors = [torch.squeeze(t, dim=self.split_dim) for t in split_tensor]
        # Return the squeezed tensors as a list
        return squeezed_tensors

# Initialize the model with the dimension along which to split
split_dim = 1  # Example: splitting along the channel dimension
model = SplitSqueezeModel(split_dim)

# Generate an input tensor of shape (1, 3, 64, 64)
input_tensor = torch.randn(1, 3, 64, 64)

# Get the output by passing the input tensor through the model
output = model(input_tensor)

# Output will be a list of squeezed tensors
for idx, tensor in enumerate(output):
    print(f"Squeezed tensor {idx}: shape {tensor.shape}")
