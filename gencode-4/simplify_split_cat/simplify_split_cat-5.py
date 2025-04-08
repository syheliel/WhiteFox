import torch

class SplitCatModel(torch.nn.Module):
    def __init__(self, split_sections, dim=0):
        super().__init__()
        self.split_sections = split_sections
        self.dim = dim

    def forward(self, x):
        # Split the input tensor into chunks
        t1 = torch.split(x, self.split_sections)
        # Access each chunk (this is just a demonstration, as we could directly use t1)
        t2 = [t1[i] for i in range(len(t1))]
        # Stack the chunks along the specified dimension
        t3 = torch.cat(t2, dim=self.dim)
        return t3

# Initializing the model with split sections and dimension
split_sections = [2, 3, 5]  # Example split sizes
model = SplitCatModel(split_sections, dim=0)

# Inputs to the model
input_tensor = torch.randn(10, 3, 64, 64)  # Batch size of 10, 3 channels, 64x64 image
output_tensor = model(input_tensor)

# Print the output shape
print("Output shape:", output_tensor.shape)
