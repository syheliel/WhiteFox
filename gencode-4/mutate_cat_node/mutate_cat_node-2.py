import torch

class SplitConcatModel(torch.nn.Module):
    def __init__(self, split_sections, split_dim):
        super().__init__()
        self.split_sections = split_sections
        self.split_dim = split_dim

    def forward(self, x):
        # Step 1: Split the input tensor into chunks
        t1 = torch.split(x, self.split_sections, dim=self.split_dim)
        
        # Step 2: Get each chunk (this is implicit in the loop)
        t2 = [t1[i] for i in range(len(t1))]
        
        # Step 3: Concatenate the chunks along the specified dimension
        t3 = torch.cat(t2, dim=self.split_dim)
        
        return t3

# Initialize the model with specified split sections and dimension
split_sections = (2, 2)  # Example split sections
split_dim = 1  # Split along dimension 1 (channels)
model = SplitConcatModel(split_sections, split_dim)

# Generate a random input tensor
input_tensor = torch.randn(1, 4, 64, 64)  # Batch size of 1, 4 channels, 64x64 spatial dimensions

# Forward pass through the model
output_tensor = model(input_tensor)
print(output_tensor.shape)  # Print the shape of the output tensor
