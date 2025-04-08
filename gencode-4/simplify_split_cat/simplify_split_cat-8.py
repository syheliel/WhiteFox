import torch

class SplitAndStackModel(torch.nn.Module):
    def __init__(self, split_sections, dim=1):
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

# Example usage
split_sections = [2, 3, 2]  # Example split sizes
model = SplitAndStackModel(split_sections)

# Generate an input tensor
input_tensor = torch.randn(1, 7, 64, 64)  # Batch size of 1, 7 channels, 64x64 images
output_tensor = model(input_tensor)

print("Output tensor shape:", output_tensor.shape)  # Display the shape of the output tensor
