import torch

class SplitCatModel(torch.nn.Module):
    def __init__(self, split_sections, dim=0):
        super().__init__()
        self.split_sections = split_sections
        self.dim = dim

    def forward(self, input_tensor):
        # Split the input tensor into chunks
        t1 = torch.split(input_tensor, self.split_sections)
        # Get each chunk of the split tensor
        t2 = [t1[i] for i in range(len(t1))]
        # Stack the chunks along the specified dimension
        t3 = torch.stack(t2, dim=self.dim)
        return t3

# Initializing the model with specified split sections
split_sections = [2, 3, 4]  # Example: split into chunks of size 2, 3, and 4
model = SplitCatModel(split_sections)

# Inputs to the model
input_tensor = torch.randn(1, 3, 9)  # Input tensor with shape (1, 3, 9)
output = model(input_tensor)

# Print output shape
print("Output shape:", output.shape)
