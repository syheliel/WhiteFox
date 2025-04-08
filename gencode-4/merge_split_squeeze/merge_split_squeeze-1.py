import torch

class SplitSqueezeModel(torch.nn.Module):
    def __init__(self, split_dim):
        super().__init__()
        self.split_dim = split_dim
    
    def forward(self, x):
        # Split the input tensor into chunks of size 1 along the specified dimension
        split_sizes = [1] * x.size(self.split_dim)  # Create a list of 1s for each element along the split dimension
        split_tensor = torch.split(x, split_sizes, dim=self.split_dim)
        
        # Squeeze each chunk along the same dimension
        squeezed_tensors = [torch.squeeze(t, dim=self.split_dim) for t in split_tensor]
        
        return squeezed_tensors

# Initialize the model with a specific dimension to split
split_dim = 1  # For example, splitting along the channel dimension
model = SplitSqueezeModel(split_dim)

# Create an input tensor (e.g., a batch of 2 images with 3 channels and 64x64 pixels)
input_tensor = torch.randn(2, 3, 64, 64)

# Run the model with the input tensor
output = model(input_tensor)

# Print the shapes of the output tensors
for i, t in enumerate(output):
    print(f"Squeezed tensor {i} shape: {t.shape}")
