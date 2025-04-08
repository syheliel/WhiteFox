import torch

# Define the model
class SplitConcatModel(torch.nn.Module):
    def __init__(self, split_sections):
        super().__init__()
        self.split_sections = split_sections

    def forward(self, x):
        # Split the input tensor into chunks
        t1 = torch.split(x, self.split_sections, dim=1)  # Splitting along dimension 1 (channels)
        # Get each chunk (this operation is not necessary but included to match the pattern)
        t2 = [t1[i] for i in range(len(t1))]
        # Concatenate the chunks along the same dimension
        t3 = torch.cat(t2, dim=1)
        return t3

# Initialize the model with specific split_sections
split_sections = (2, 2)  # Example split sizes for the channel dimension
model = SplitConcatModel(split_sections)

# Generate an input tensor for the model
input_tensor = torch.randn(1, 4, 64, 64)  # Shape: (batch_size, channels, height, width)

# Forward pass through the model
output_tensor = model(input_tensor)

# Print the output tensor shape
print(output_tensor.shape)
