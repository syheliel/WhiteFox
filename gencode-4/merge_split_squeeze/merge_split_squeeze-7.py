import torch

# Model definition
class SplitSqueezeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        # Split the input tensor into chunks of size 1 along dimension 1
        split_sizes = [1] * input_tensor.size(1)  # Create a list of 1's based on the number of channels
        split_tensor = torch.split(input_tensor, split_sizes, dim=1)
        
        # Squeeze each chunk of the split tensor along the same dimension
        squeezed_tensors = [torch.squeeze(t, dim=1) for t in split_tensor]
        
        return squeezed_tensors

# Initializing the model
model = SplitSqueezeModel()

# Creating an input tensor
# Here, we create a tensor of shape (1, 3, 64, 64), meaning batch size of 1, 3 channels, and 64x64 spatial dimensions
input_tensor = torch.randn(1, 3, 64, 64)

# Run the model
output_tensors = model(input_tensor)

# Print the output shapes
for i, tensor in enumerate(output_tensors):
    print(f"Output tensor {i} shape: {tensor.shape}")
