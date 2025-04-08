import torch

class SplitConcatModel(torch.nn.Module):
    def __init__(self, split_sections, dim):
        super().__init__()
        self.split_sections = split_sections
        self.dim = dim

    def forward(self, x):
        # Split the input tensor into chunks based on split_sections
        t1 = torch.split(x, self.split_sections, dim=self.dim)
        
        # Get each chunk (this step is somewhat redundant here; we are just collecting them)
        t2 = [t1[i] for i in range(len(t1))]
        
        # Concatenate the chunks along the specified dimension
        t3 = torch.cat(t2, dim=self.dim)
        
        return t3

# Initializing the model with specific split_sections and dimension
split_sections = (2, 2, 2)  # Example split sizes
dim = 1  # Dimension along which to split and concatenate
model = SplitConcatModel(split_sections, dim)

# Generating input tensor
input_tensor = torch.randn(1, 6, 64)  # Example input tensor with shape (1, 6, 64)

# Forward pass through the model
output_tensor = model(input_tensor)

print("Input Tensor Shape:", input_tensor.shape)
print("Output Tensor Shape:", output_tensor.shape)
