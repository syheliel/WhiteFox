import torch

class SplitConcatModel(torch.nn.Module):
    def __init__(self, split_sections, dim=0):
        super().__init__()
        self.split_sections = split_sections
        self.dim = dim

    def forward(self, x):
        # Split the input tensor into chunks according to split_sections
        t1 = torch.split(x, self.split_sections)
        
        # Access each chunk (this is just done implicitly here)
        t2 = [t1[i] for i in range(len(t1))]
        
        # Stack the chunks along the specified dimension
        t3 = torch.stack(t2, dim=self.dim)
        
        return t3

# Initializing the model with specific split sections
split_sections = [2, 2]  # Example split sizes
model = SplitConcatModel(split_sections, dim=0)

# Inputs to the model
input_tensor = torch.randn(4, 3, 64, 64)  # A random tensor with shape (4, 3, 64, 64)
output_tensor = model(input_tensor)

# Displaying the output
print(output_tensor.shape)  # Output shape will depend on the split sizes
