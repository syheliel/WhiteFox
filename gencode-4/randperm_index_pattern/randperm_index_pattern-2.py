import torch

class ShuffleModel(torch.nn.Module):
    def __init__(self, slice_shape):
        super(ShuffleModel, self).__init__()
        self.slice_shape = slice_shape

    def forward(self, x):
        index = torch.randperm(x.shape[0], device=x.device)[:self.slice_shape]  # Generate a random permutation of integers
        output = x[index]  # Index the input tensor x with the generated index
        return output, index  # Return the shuffled tensor and the indices used for shuffling

# Initialize the model with a specific slice_shape
slice_shape = 5  # For example, we want to get 5 elements from the tensor
model = ShuffleModel(slice_shape)

# Create an input tensor (e.g., with shape [10, 3, 64, 64])
input_tensor = torch.randn(10, 3, 64, 64)

# Get the output from the model
shuffled_output, indices = model(input_tensor)

# Print the shapes of the output and indices
print("Shuffled Output Shape:", shuffled_output.shape)  # Should be [slice_shape, 3, 64, 64]
print("Indices Used for Shuffling:", indices)  # Should be a tensor of shape [slice_shape]
