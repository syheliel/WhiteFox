import torch

# Model definition
class SplitConcatModel(torch.nn.Module):
    def __init__(self, split_sections, split_dim):
        super().__init__()
        self.split_sections = split_sections
        self.split_dim = split_dim

    def forward(self, input_tensor):
        # Split the input tensor into chunks
        t1 = torch.split(input_tensor, self.split_sections, dim=self.split_dim)
        
        # Get each chunk (this step is redundant since t1 already contains the chunks)
        t2 = [t1[i] for i in range(len(t1))]
        
        # Concatenate the chunks along the specified dimension
        t3 = torch.cat(t2, dim=self.split_dim)
        return t3

# Define the splitting arguments
split_sections = (2, 2, 2)  # Example split sizes for a tensor with 6 elements in the specified dimension
split_dim = 1  # We'll split along the second dimension

# Initializing the model
model = SplitConcatModel(split_sections, split_dim)

# Inputs to the model
input_tensor = torch.randn(1, 6, 64, 64)  # A tensor of shape (1, 6, 64, 64)
output = model(input_tensor)

# Display the output shape
print("Output shape:", output.shape)
