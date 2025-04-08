import torch

class SplitConcatModel(torch.nn.Module):
    def __init__(self, split_sections):
        super().__init__()
        self.split_sections = split_sections

    def forward(self, x):
        # Split the input tensor into chunks
        t1 = torch.split(x, self.split_sections, dim=1)  # Assume we are splitting along dim=1
        # Get each chunk
        t2 = [t1[i] for i in range(len(t1))]
        # Concatenate the chunks along the same dimension
        t3 = torch.cat(t2, dim=1)
        return t3

# Example usage
split_sections = (2, 2, 2)  # Example split sections for the second dimension
model = SplitConcatModel(split_sections)

# Input tensor
input_tensor = torch.randn(1, 6, 64, 64)  # Batch size of 1, 6 channels, 64x64 image
output_tensor = model(input_tensor)

print("Output tensor shape:", output_tensor.shape)
