import torch

# Define the Model class
class Model(torch.nn.Module):
    def __init__(self, split_sections):
        super().__init__()
        self.split_sections = split_sections

    def forward(self, x1):
        # Split the input tensor into chunks
        t1 = torch.split(x1, self.split_sections, dim=1)
        # Retrieve each chunk
        t2 = [t1[i] for i in range(len(t1))]
        # Concatenate the chunks along the specified dimension
        t3 = torch.cat(t2, dim=1)
        return t3

# Initializing the model with specific split sections
split_sections = (1, 1, 1)  # Example sections for a tensor with 3 channels
m = Model(split_sections)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)  # Input tensor of shape (batch_size, channels, height, width)

# Forward pass through the model
output = m(x1)

# Output shape
print(output.shape)  # Should be the same as input shape: (1, 3, 64, 64)
