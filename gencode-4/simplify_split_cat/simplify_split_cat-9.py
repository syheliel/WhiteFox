import torch

class SplitCatModel(torch.nn.Module):
    def __init__(self, split_sections, dim):
        super().__init__()
        self.split_sections = split_sections
        self.dim = dim

    def forward(self, x):
        # Split the input tensor into chunks
        t1 = torch.split(x, self.split_sections)
        # Get each chunk of the split tensor
        t2 = [t1[i] for i in range(len(t1))]
        # Stack the chunks along the specified dimension
        t3 = torch.cat(t2, dim=self.dim)  # Using concatenate here instead of stack
        return t3

# Example usage
split_sections = [16, 16, 16, 16]  # Splitting the input tensor into chunks of varying sizes
dim = 1  # Concatenating along dimension 1

# Initializing the model
model = SplitCatModel(split_sections, dim)

# Inputs to the model
input_tensor = torch.randn(1, 64, 64)  # A random input tensor of shape (1, 64, 64)

# Forward pass
output_tensor = model(input_tensor)

print(f"Output tensor shape: {output_tensor.shape}")
