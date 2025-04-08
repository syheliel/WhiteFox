import torch

# Model definition
class ShuffleModel(torch.nn.Module):
    def __init__(self, slice_shape):
        super().__init__()
        self.slice_shape = slice_shape

    def forward(self, x):
        # Generate a random permutation of integers from 0 to x.shape[0]
        index = torch.randperm(x.shape[0], device=x.device)[:self.slice_shape]
        # Index the input tensor x with the generated index
        output = x[index]
        return output, index

# Example parameters
slice_shape = 5  # Define the shape/size of the shuffled subset

# Initialize the model
model = ShuffleModel(slice_shape)

# Create an input tensor
# Example input tensor of shape (10, 3, 64, 64) representing a batch of 10 images with 3 channels of size 64x64
input_tensor = torch.randn(10, 3, 64, 64)

# Get the output of the model
output_tensor, indices = model(input_tensor)

# Print the outputs for verification
print("Output Tensor Shape:", output_tensor.shape)
print("Indices Used for Shuffling:", indices)
