import torch

class SplitConcatModel(torch.nn.Module):
    def __init__(self, split_sections, indices, concat_dim):
        super().__init__()
        self.split_sections = split_sections
        self.indices = indices
        self.concat_dim = concat_dim

    def forward(self, x):
        # Split the input tensor into chunks along the specified dimension
        split_tensors = torch.split(x, self.split_sections, dim=self.concat_dim)
        
        # Select certain chunks from the split tensor
        selected_chunks = [split_tensors[i] for i in self.indices]
        
        # Concatenate the selected chunks along the same dimension
        concatenated = torch.cat(selected_chunks, dim=self.concat_dim)
        
        return concatenated

# Parameters for the model
split_sections = [2, 2, 2]  # Example split sections
indices = [0, 1]  # Example indices to select
concat_dim = 1  # Dimension along which to split and concatenate

# Initializing the model
model = SplitConcatModel(split_sections, indices, concat_dim)

# Inputs to the model
input_tensor = torch.randn(1, 6, 64, 64)  # Batch size of 1, 6 channels, 64x64 spatial dimensions
output_tensor = model(input_tensor)

# Print the shape of the output tensor
print(output_tensor.shape)
