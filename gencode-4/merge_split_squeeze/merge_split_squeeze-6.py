import torch

class SplitSqueezeModel(torch.nn.Module):
    def __init__(self, split_size):
        super().__init__()
        self.split_size = split_size  # The sizes to split the input tensor

    def forward(self, x):
        # Split the input tensor along the specified dimension
        split_tensor = torch.split(x, self.split_size, dim=1)  # Split along the channel dimension
        # Squeeze each chunk of the split tensor along the same dimension
        squeezed_tensors = [torch.squeeze(t, dim=1) for t in split_tensor]
        return squeezed_tensors

# Initializing the model with a split size of 1
model = SplitSqueezeModel(split_size=1)

# Generating inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Example input tensor with shape (batch_size, channels, height, width)
output = model(input_tensor)

# Print output shapes to verify
for i, tensor in enumerate(output):
    print(f'Squeezed tensor {i} shape: {tensor.shape}')
