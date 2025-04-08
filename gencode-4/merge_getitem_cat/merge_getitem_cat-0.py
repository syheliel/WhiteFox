import torch

# Define the model that follows the specified pattern
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Split the input tensor into 4 chunks along the specified dimension (dim=1)
        t1 = torch.split(x, split_sections=1, dim=1)  # Splitting along the channel dimension
        # Select the first and third chunks from the split tensor (indices 0 and 2)
        indices = [0, 2]
        t2 = [t1[i] for i in indices]
        # Concatenate the selected chunks along the same dimension
        t3 = torch.cat(t2, dim=1)
        return t3

# Initializing the model
model = Model()

# Generate an input tensor of shape (1, 4, 64, 64) which has 4 channels
input_tensor = torch.randn(1, 4, 64, 64)

# Forward pass through the model
output_tensor = model(input_tensor)

# Print the shape of the output tensor
print("Input tensor shape:", input_tensor.shape)
print("Output tensor shape:", output_tensor.shape)
