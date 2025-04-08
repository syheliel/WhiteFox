import torch

class SplitCatModel(torch.nn.Module):
    def __init__(self, split_sections, dim=0):
        super().__init__()
        self.split_sections = split_sections
        self.dim = dim
    
    def forward(self, input_tensor):
        # Split the input tensor into chunks
        chunks = torch.split(input_tensor, self.split_sections)
        # Access each chunk (this step is somewhat redundant, but included as per requirements)
        chunk_list = [chunks[i] for i in range(len(chunks))]
        
        # Stack the chunks along the specified dimension
        output_tensor = torch.stack(chunk_list, dim=self.dim)
        return output_tensor

# Initialize the model with specific split sizes and dimension
split_sizes = [2, 3, 5]  # Example split sizes
model = SplitCatModel(split_sections=split_sizes, dim=0)

# Inputs to the model
input_tensor = torch.randn(10, 10)  # Example input tensor of size (10, 10)

# Forward pass
output_tensor = model(input_tensor)
print(output_tensor)
