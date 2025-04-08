import torch

class SplitSqueezeModel(torch.nn.Module):
    def __init__(self, split_dim=1):
        super().__init__()
        self.split_dim = split_dim

    def forward(self, input_tensor):
        # Split the input tensor along the specified dimension
        split_sizes = [1] * input_tensor.size(self.split_dim)  # All sizes are 1
        split_tensor = torch.split(input_tensor, split_sizes, dim=self.split_dim)

        # Squeeze each chunk along the same dimension
        squeezed_tensors = [torch.squeeze(t, dim=self.split_dim) for t in split_tensor]
        
        # Return the squeezed tensors as a list
        return squeezed_tensors

# Initialize the model
model = SplitSqueezeModel(split_dim=1)

# Create an input tensor of shape (2, 3, 4, 4)
input_tensor = torch.randn(2, 3, 4, 4)

# Get the output from the model
output = model(input_tensor)

# Print the output shapes for verification
print([t.shape for t in output])

input_tensor = torch.randn(2, 3, 4, 4)  # Example input tensor with shape (batch_size, channels, height, width)
