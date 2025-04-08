import torch

class SplitConcatModel(torch.nn.Module):
    def __init__(self, split_sections, selected_indices):
        super().__init__()
        self.split_sections = split_sections
        self.selected_indices = selected_indices
    
    def forward(self, x):
        # Split the input tensor along the specified dimension (dim=1)
        t1 = torch.split(x, self.split_sections, dim=1)
        
        # Select certain chunks from the split tensor
        t2 = [t1[i] for i in self.selected_indices]
        
        # Concatenate the selected chunks along the same dimension
        t3 = torch.cat(t2, dim=1)
        
        return t3

# Initialize the model with specific split sections and selected indices
split_sections = [1, 2, 1]  # Example split sizes along dimension 1
selected_indices = [0, 2]    # We want to select the first and third chunks

model = SplitConcatModel(split_sections, selected_indices)

# Generate an input tensor
input_tensor = torch.randn(1, 4, 64, 64)  # Batch size of 1, 4 channels, 64x64 spatial dimensions

# Forward pass through the model
output = model(input_tensor)

# Print output shape
print("Output shape:", output.shape)
