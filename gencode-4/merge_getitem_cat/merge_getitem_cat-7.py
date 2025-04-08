import torch

class ChunkSelectorModel(torch.nn.Module):
    def __init__(self, split_sections, indices):
        super().__init__()
        self.split_sections = split_sections
        self.indices = indices

    def forward(self, x):
        # Split the input tensor into chunks along the specified dimension
        t1 = torch.split(x, self.split_sections, dim=1)  # Split along channel dimension (dim=1)
        
        # Select certain chunks from the split tensor based on indices
        t2 = [t1[i] for i in self.indices]  # Select only the chunks specified by indices
        
        # Concatenate the selected chunks along the same dimension
        t3 = torch.cat(t2, dim=1)  # Concatenate along channel dimension (dim=1)
        
        return t3

# Specify the parameters for splitting and selecting
split_sections = 4  # Number of sections to split each tensor along the channel dimension
indices = [0, 2]    # Indices of the chunks to select

# Initialize the model
model = ChunkSelectorModel(split_sections, indices)

# Create an input tensor with shape (batch_size, channels, height, width)
input_tensor = torch.randn(1, 8, 64, 64)  # Batch size 1, 8 channels, 64x64 dimensions

# Obtain the output from the model
output_tensor = model(input_tensor)

print("Input tensor shape:", input_tensor.shape)
print("Output tensor shape:", output_tensor.shape)
