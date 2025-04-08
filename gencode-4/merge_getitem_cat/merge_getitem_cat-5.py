import torch

class Model(torch.nn.Module):
    def __init__(self, split_sections, indices):
        super().__init__()
        self.split_sections = split_sections
        self.indices = indices

    def forward(self, x):
        # Split the input tensor into chunks along dimension 1
        t1 = torch.split(x, self.split_sections, dim=1)
        # Select certain chunks from the split tensor
        t2 = [t1[i] for i in self.indices]
        # Concatenate the selected chunks along dimension 1
        t3 = torch.cat(t2, dim=1)
        return t3

# Initializing the model
# Assuming we want to split the input tensor into chunks of size 2 along dimension 1
split_sections = 2  # Size of each chunk
indices = [0, 1]    # Select the first two chunks (in this case, both will be included)
m = Model(split_sections, indices)

# Inputs to the model
# Creating a random input tensor with shape (1, 6, 64, 64)
# Here, the second dimension size must be a multiple of the split_sections for proper splitting
x1 = torch.randn(1, 6, 64, 64)

# Getting the output from the model
output = m(x1)

# Display the shape of the output
print(output.shape)
