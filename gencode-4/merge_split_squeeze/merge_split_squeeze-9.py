import torch

# Model definition
class SplitSqueezeModel(torch.nn.Module):
    def __init__(self, split_dim):
        super().__init__()
        self.split_dim = split_dim

    def forward(self, input_tensor):
        # Split the input tensor into chunks of size 1 along the specified dimension
        split_sizes = [1] * input_tensor.size(self.split_dim)  # All elements are 1
        split_tensor = torch.split(input_tensor, split_sizes, dim=self.split_dim)
        # Squeeze each chunk of the split tensor along the same dimension
        squeezed_tensors = [torch.squeeze(t, dim=self.split_dim) for t in split_tensor]
        return squeezed_tensors

# Initializing the model
split_dim = 1  # Choose dimension to split
model = SplitSqueezeModel(split_dim)

# Inputs to the model
input_tensor = torch.randn(4, 3, 64, 64)  # Example input tensor with shape (4, 3, 64, 64)
output_tensors = model(input_tensor)

# Check the output shapes
for i, tensor in enumerate(output_tensors):
    print(f"Squeezed tensor {i} shape: {tensor.shape}")
