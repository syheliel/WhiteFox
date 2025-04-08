import torch

class SplitCatModel(torch.nn.Module):
    def __init__(self, split_sections, dim=0):
        super().__init__()
        self.split_sections = split_sections
        self.dim = dim

    def forward(self, x):
        # Split the input tensor into chunks
        split_tensors = torch.split(x, self.split_sections)
        
        # Access each chunk of the split tensor
        chunks = [split_tensors[i] for i in range(len(split_tensors))]
        
        # Stack the chunks along the specified dimension
        output = torch.stack(chunks, dim=self.dim)
        
        return output

# Initializing the model with specific split sections
split_sections = [2, 2]  # Example split sizes
model = SplitCatModel(split_sections=split_sections, dim=0)

# Generating input tensor
input_tensor = torch.randn(4, 3, 64, 64)  # Shape: [batch_size, channels, height, width]

# Getting the output from the model
output_tensor = model(input_tensor)

print("Output tensor shape:", output_tensor.shape)
