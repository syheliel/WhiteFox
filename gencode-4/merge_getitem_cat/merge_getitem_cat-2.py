import torch

# Model definition
class SplitSelectConcatModel(torch.nn.Module):
    def __init__(self, split_sections):
        super().__init__()
        self.split_sections = split_sections

    def forward(self, x):
        t1 = torch.split(x, self.split_sections, dim=1)  # Split the input tensor along dimension 1
        indices = [0, 1]  # Selecting the first two chunks
        t2 = [t1[i] for i in indices]  # Select certain chunks from the split tensor
        t3 = torch.cat(t2, dim=1)  # Concatenate the selected chunks along dimension 1
        return t3

# Initializing the model with split sections
split_sections = 2  # Assuming we want to split the input tensor into 2 sections along dimension 1
model = SplitSelectConcatModel(split_sections)

# Inputs to the model
input_tensor = torch.randn(1, 4, 64, 64)  # Batch size of 1, 4 channels, 64x64 spatial dimensions
output_tensor = model(input_tensor)

# Print the output shape for verification
print(output_tensor.shape)  # Should be [1, 2, 64, 64] since we are concatenating the first two channels
