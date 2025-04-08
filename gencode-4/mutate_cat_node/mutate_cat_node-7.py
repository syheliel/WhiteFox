import torch

# Model
class SplitConcatModel(torch.nn.Module):
    def __init__(self, split_sections, split_dim):
        super().__init__()
        self.split_sections = split_sections
        self.split_dim = split_dim

    def forward(self, input_tensor):
        # Split the input tensor into chunks along the specified dimension
        t1 = torch.split(input_tensor, self.split_sections, dim=self.split_dim)
        
        # Get each chunk
        t2 = [t1[i] for i in range(len(t1))]
        
        # Concatenate the chunks along the specified dimension
        t3 = torch.cat(t2, dim=self.split_dim)
        
        return t3

# Initializing the model
split_sections = (2, 3, 2)  # Example split sizes
split_dim = 1  # Dimension to split along
model = SplitConcatModel(split_sections, split_dim)

# Input tensor for the model
input_tensor = torch.randn(5, 9)  # Shape compatible with the split sections (5 rows, 9 columns)

# Forward pass
output = model(input_tensor)

# Print the output
print("Output shape:", output.shape)
