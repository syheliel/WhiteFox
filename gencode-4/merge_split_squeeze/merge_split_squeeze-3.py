import torch

# Model definition
class SplitSqueezeModel(torch.nn.Module):
    def __init__(self, split_dim):
        super().__init__()
        self.split_dim = split_dim  # Dimension to split along

    def forward(self, x):
        # Split the input tensor into chunks of size 1 along the specified dimension
        split_tensor = torch.split(x, 1, dim=self.split_dim)
        # Squeeze each chunk of the split tensor along the same dimension
        squeezed_tensors = [torch.squeeze(t, dim=self.split_dim) for t in split_tensor]
        return squeezed_tensors

# Initializing the model with a specified dimension to split
split_dim = 1  # For example, splitting along the channel dimension
model = SplitSqueezeModel(split_dim)

# Inputs to the model
input_tensor = torch.randn(2, 3, 64, 64)  # Batch size of 2, 3 channels, 64x64 spatial dimensions
output_tensors = model(input_tensor)

# Display the shapes of the output tensors
for i, tensor in enumerate(output_tensors):
    print(f"Output tensor {i} shape: {tensor.shape}")
