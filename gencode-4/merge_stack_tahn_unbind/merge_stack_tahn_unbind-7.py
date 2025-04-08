import torch

class SplitStackTanhModel(torch.nn.Module):
    def __init__(self, split_sections):
        super().__init__()
        self.split_sections = split_sections

    def forward(self, x):
        # Split the input tensor into chunks along the specified dimension
        t1 = torch.split(x, self.split_sections, dim=1)  # Assume input tensor has channels along dim=1
        # Stack the chunks along a new dimension
        t2 = torch.stack(t1, dim=2)  # Stack along a new dimension (e.g., the 3rd dimension)
        # Apply the hyperbolic tangent function
        t3 = torch.tanh(t2)
        return t3

# Initializing the model with split sections
split_sections = 2  # Example to split the input tensor into 2 sections
model = SplitStackTanhModel(split_sections)

# Inputs to the model
input_tensor = torch.randn(1, 4, 64, 64)  # Input shape: (batch_size, channels, height, width)
output = model(input_tensor)

print("Output shape:", output.shape)
