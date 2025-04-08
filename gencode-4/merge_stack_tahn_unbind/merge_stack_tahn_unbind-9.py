import torch

# Model Definition
class CustomModel(torch.nn.Module):
    def __init__(self, split_sections):
        super().__init__()
        self.split_sections = split_sections

    def forward(self, input_tensor):
        # Split the input tensor into chunks along the specified dimension
        t1 = torch.split(input_tensor, self.split_sections, dim=1)
        # Stack the chunks along a new dimension
        t2 = torch.stack(t1, dim=2)
        # Apply the hyperbolic tangent function to the stacked tensor
        t3 = torch.tanh(t2)
        return t3

# Initializing the model with specified split sections
split_sections = 2  # This means we will split the input tensor along dimension 1 into chunks of size 2
model = CustomModel(split_sections)

# Generating input tensor
input_tensor = torch.randn(1, 4, 64, 64)  # Input shape (batch_size, channels, height, width)

# Forward pass through the model
output = model(input_tensor)

# Print the output shape
print("Output shape:", output.shape)
