import torch

class ChunkModel(torch.nn.Module):
    def __init__(self, split_sections, indices):
        super().__init__()
        self.split_sections = split_sections
        self.indices = indices

    def forward(self, input_tensor):
        # Split the input tensor into chunks along the specified dimension
        t1 = torch.split(input_tensor, self.split_sections, dim=1)  # Splitting along dimension 1
        # Select certain chunks from the split tensor
        t2 = [t1[i] for i in self.indices]
        # Concatenate the selected chunks along the same dimension
        t3 = torch.cat(t2, dim=1)  # Concatenating along dimension 1
        return t3

# Example usage
split_sections = 2  # Example split size (2 channels each)
indices = [0, 1]    # Selecting both chunks in this case

# Initialize the model
model = ChunkModel(split_sections, indices)

# Generate a random input tensor (e.g., with 4 channels and height/width of 64)
input_tensor = torch.randn(1, 4, 64, 64)  # Batch size of 1, 4 channels

# Forward pass through the model
output = model(input_tensor)

# Output shape
print("Output shape:", output.shape)
