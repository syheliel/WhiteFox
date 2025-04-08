import torch

class SplitConcatModel(torch.nn.Module):
    def __init__(self, split_sections, indices, dim):
        super().__init__()
        self.split_sections = split_sections
        self.indices = indices
        self.dim = dim

    def forward(self, x):
        # Split the input tensor into chunks along the specified dimension
        split_tensors = torch.split(x, self.split_sections, dim=self.dim)
        
        # Select certain chunks from the split tensor
        selected_chunks = [split_tensors[i] for i in self.indices]
        
        # Concatenate the selected chunks along the same dimension
        concatenated_tensor = torch.cat(selected_chunks, dim=self.dim)
        
        return concatenated_tensor

# Initialize the model with specific split sections and indices
split_sections = 16  # Each chunk will have a size of 16
indices = [0, 1, 2]  # Selecting the first three chunks
dim = 1  # Split and concatenate along the channel dimension

# Create an instance of the model
model = SplitConcatModel(split_sections, indices, dim)

# Generate an input tensor of size [1, 48, 64, 64] (batch_size, channels, height, width)
input_tensor = torch.randn(1, 48, 64, 64)

# Pass the input tensor through the model
output_tensor = model(input_tensor)

print("Output tensor shape:", output_tensor.shape)  # Should reflect the concatenated result
