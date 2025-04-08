import torch

class SplitConcatModel(torch.nn.Module):
    def __init__(self, split_sections, dim=0):
        super(SplitConcatModel, self).__init__()
        self.split_sections = split_sections
        self.dim = dim

    def forward(self, input_tensor):
        # Split the input tensor into chunks
        t1 = torch.split(input_tensor, self.split_sections)
        # Get each chunk of the split tensor (this is redundant, but simulates the pattern)
        t2 = [t1[i] for i in range(len(t1))]
        # Stack the chunks along the specified dimension
        t3 = torch.stack(t2, dim=self.dim)
        return t3

# Initialize the model with split sections and dimension
split_sections = [2, 3, 5]  # Example split sizes
model = SplitConcatModel(split_sections, dim=0)

# Generate input tensor
input_tensor = torch.randn(1, 3, 10)  # Batch size of 1, 3 channels, 10 elements
output = model(input_tensor)

print("Input Tensor Shape:", input_tensor.shape)
print("Output Tensor Shape:", output.shape)
