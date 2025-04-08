import torch

class SplitCatModel(torch.nn.Module):
    def __init__(self, split_sections, dim=0):
        super().__init__()
        self.split_sections = split_sections
        self.dim = dim

    def forward(self, input_tensor):
        # Split the input tensor into chunks
        t1 = torch.split(input_tensor, self.split_sections)
        # Access each chunk (this is effectively done implicitly)
        t2 = [t1[i] for i in range(len(t1))]
        # Stack the chunks along the specified dimension
        t3 = torch.stack(t2, dim=self.dim)
        return t3

# Initializing the model with specified split sections and dimension
split_sections = [2, 3, 5]  # Example lengths for each chunk
model = SplitCatModel(split_sections, dim=0)

# Generating an input tensor for the model
input_tensor = torch.randn(10, 3, 64, 64)  # Random input tensor of shape (10, 3, 64, 64)

# Getting the output from the model
output = model(input_tensor)

print(output.shape)  # Print the shape of the output tensor
