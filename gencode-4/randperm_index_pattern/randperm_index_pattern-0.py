import torch

class ShuffleModel(torch.nn.Module):
    def __init__(self, slice_shape):
        super().__init__()
        self.slice_shape = slice_shape

    def forward(self, x):
        index = torch.randperm(x.shape[0], device=x.device)[:self.slice_shape]  # Generate random permutation
        output = x[index]  # Index the input tensor x with the generated index
        return output, index

# Initialize the model with a specified slice shape
slice_shape = 5  # Example slice shape
model = ShuffleModel(slice_shape)

# Example input tensor
batch_size = 10
channels = 3
height = 64
width = 64
x_input = torch.randn(batch_size, channels, height, width)

# Get the output from the model
output_tensor, indices_used = model(x_input)

# Print the shapes of the output and indices
print("Output tensor shape:", output_tensor.shape)
print("Indices used for shuffling:", indices_used)
