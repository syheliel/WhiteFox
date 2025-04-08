import torch

class SplitCatModel(torch.nn.Module):
    def __init__(self, split_sections, dim=1):
        super().__init__()
        self.split_sections = split_sections
        self.dim = dim

    def forward(self, x):
        # Split the input tensor into chunks
        t1 = torch.split(x, self.split_sections)
        # Get each chunk of the split tensor (this actually does nothing but demonstrates the pattern)
        t2 = [t1[i] for i in range(len(t1))]
        # Stack the chunks along the specified dimension
        t3 = torch.stack(t2, dim=self.dim)
        return t3

# Initializing the model
split_sections = [2, 3, 4]  # Example split sizes
model = SplitCatModel(split_sections, dim=0)

# Inputs to the model
input_tensor = torch.randn(9, 3, 64, 64)  # A tensor with shape (9, 3, 64, 64)
output_tensor = model(input_tensor)

# Output shape
print("Output shape:", output_tensor.shape)
