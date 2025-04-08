import torch

# Model definition
class SplitConcatModel(torch.nn.Module):
    def __init__(self, split_sections, split_dim):
        super().__init__()
        self.split_sections = split_sections
        self.split_dim = split_dim

    def forward(self, x):
        # Split the input tensor into chunks
        t1 = torch.split(x, self.split_sections, dim=self.split_dim)
        # Get each chunk (this is essentially redundant, but keeps in line with the pattern)
        t2 = [t1[i] for i in range(len(t1))]
        # Concatenate the chunks along the specified dimension
        t3 = torch.cat(t2, dim=self.split_dim)
        return t3

# Initializing the model with specific split sections and dimension
split_sections = (32, 32)  # Example of splitting a dimension into two chunks of size 32
split_dim = 1              # Choosing to split along dimension 1
model = SplitConcatModel(split_sections, split_dim)

# Inputs to the model
input_tensor = torch.randn(1, 64, 64)  # Example input tensor of shape (1, 64, 64)
output_tensor = model(input_tensor)

# Display the output shape
print(output_tensor.shape)
