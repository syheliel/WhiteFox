import torch

# Define the model
class SplitSqueezeModel(torch.nn.Module):
    def __init__(self, split_size):
        super().__init__()
        self.split_size = split_size  # Size of each split

    def forward(self, input_tensor):
        # Split the input tensor into chunks of size `split_size` along the specified dimension
        split_tensor = torch.split(input_tensor, self.split_size, dim=1)  # Splitting along dimension 1
        # Squeeze each chunk of the split tensor along the same dimension
        squeezed_tensors = [torch.squeeze(t, dim=1) for t in split_tensor]
        return squeezed_tensors

# Initialize the model with a split size of 1
model = SplitSqueezeModel(split_size=1)

# Create an input tensor with shape (1, 4, 64, 64)
input_tensor = torch.randn(1, 4, 64, 64)

# Pass the input tensor through the model
output = model(input_tensor)

# Print the shapes of the output tensors
for i, out in enumerate(output):
    print(f"Output tensor {i} shape: {out.shape}")
