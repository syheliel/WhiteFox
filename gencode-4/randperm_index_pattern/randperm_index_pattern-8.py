import torch

# Define the model
class ShuffleModel(torch.nn.Module):
    def __init__(self, slice_shape):
        super().__init__()
        self.slice_shape = slice_shape

    def forward(self, x):
        # Generate a random permutation of integers from 0 to x.shape[0]
        index = torch.randperm(x.shape[0], device=x.device)[:self.slice_shape]
        # Index the input tensor x with the generated index
        output = torch.index_select(x, 0, index)
        return output, index

# Initialize parameters
slice_shape = 5  # Define the size of the shuffled subset
model = ShuffleModel(slice_shape)

# Create an input tensor of shape (10, 3, 64, 64)
input_tensor = torch.randn(10, 3, 64, 64)

# Get the output from the model
output_tensor, indices = model(input_tensor)

# Print the shapes for verification
print("Output tensor shape:", output_tensor.shape)  # Should be (5, 3, 64, 64)
print("Indices used for shuffling:", indices)
