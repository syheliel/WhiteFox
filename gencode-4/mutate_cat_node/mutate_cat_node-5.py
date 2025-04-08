import torch

# Model definition
class ChunkConcatModel(torch.nn.Module):
    def __init__(self, split_sections, split_dim):
        super().__init__()
        self.split_sections = split_sections
        self.split_dim = split_dim

    def forward(self, x):
        # Split the input tensor into chunks
        t1 = torch.split(x, self.split_sections, dim=self.split_dim)
        # Get each chunk (this step is redundant but follows the specified pattern)
        t2 = [t1[i] for i in range(len(t1))]
        # Concatenate the chunks along the specified dimension
        t3 = torch.cat(t2, dim=self.split_dim)
        return t3

# Parameters for the model
split_sections = [2, 2, 2]  # Example split sizes
split_dim = 1  # Example dimension to split along

# Initializing the model
model = ChunkConcatModel(split_sections, split_dim)

# Inputs to the model
input_tensor = torch.randn(1, 6, 64)  # Batch size of 1, 6 channels, 64 features
output = model(input_tensor)

# Output shape for verification
print("Output shape:", output.shape)
