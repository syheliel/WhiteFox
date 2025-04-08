import torch

class SplitConcatModel(torch.nn.Module):
    def __init__(self, split_sections):
        super(SplitConcatModel, self).__init__()
        self.split_sections = split_sections  # List or tuple of sizes to split the input tensor

    def forward(self, input_tensor):
        # Split the input tensor into chunks along the specified dimension (dim=1)
        t1 = torch.split(input_tensor, self.split_sections, dim=1)
        
        # Retrieve each chunk (this is essentially redundant but included for clarity)
        t2 = [t1[i] for i in range(len(t1))]
        
        # Concatenate the chunks along the same dimension (dim=1)
        t3 = torch.cat(t2, dim=1)
        
        return t3

# Initializing the model with split sections
split_sections = [1, 2, 1]  # Example split sizes for a 1D split along the channel dimension
model = SplitConcatModel(split_sections)

# Generating an input tensor (e.g., a batch of images with 4 channels and size 64x64)
input_tensor = torch.randn(1, 4, 64, 64)  # Batch size of 1, 4 channels, 64x64 spatial dimensions

# Forward pass
output = model(input_tensor)

# Output shape
print(output.shape)
