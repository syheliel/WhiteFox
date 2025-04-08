import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor, split_sections):
        # Split the input tensor into chunks along dimension 1
        t1 = torch.split(input_tensor, split_sections, dim=1)
        # Stack the chunks along a new dimension (dimension 2)
        t2 = torch.stack(t1, dim=2)
        # Apply the hyperbolic tangent function
        t3 = torch.tanh(t2)
        return t3

# Initializing the model
model = Model()

# Define the input tensor and split_sections
input_tensor = torch.randn(1, 6, 64, 64)  # Batch size of 1, 6 channels, 64x64 images
split_sections = 2  # Split the tensor into 2 chunks along the channel dimension

# Get the output from the model
output = model(input_tensor, split_sections)

# Print the output shape
print(output.shape)  # Should be (1, 2, 3, 64, 64) based on the split and stacking
