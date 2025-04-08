import torch

# Model Definition
class CustomModel(torch.nn.Module):
    def __init__(self, split_sections, indices):
        super().__init__()
        self.split_sections = split_sections
        self.indices = indices

    def forward(self, x):
        # Splitting the input tensor
        t1 = torch.split(x, self.split_sections, dim=1)  # Split along dimension 1
        # Selecting specific chunks
        t2 = [t1[i] for i in self.indices]
        # Concatenating the selected chunks
        t3 = torch.cat(t2, dim=1)  # Concatenate along the same dimension
        return t3

# Parameters for splitting and selecting
split_sections = [1, 1, 1, 1]  # Example split sizes
indices = [0, 2]  # Select the first and the third chunk

# Initializing the model
model = CustomModel(split_sections, indices)

# Inputs to the model
x1 = torch.randn(1, sum(split_sections), 64, 64)  # Input tensor with a batch size of 1 and appropriate channels
output = model(x1)

print("Output shape:", output.shape)
