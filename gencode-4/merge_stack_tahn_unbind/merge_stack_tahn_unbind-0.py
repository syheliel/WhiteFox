import torch

class SplitTanhModel(torch.nn.Module):
    def __init__(self, split_sections):
        super().__init__()
        self.split_sections = split_sections

    def forward(self, x):
        # Step 1: Split the input tensor into chunks along the given dimension
        t1 = torch.split(x, self.split_sections, dim=1)  # Split along the channel dimension (dim=1)
        
        # Step 2: Stack the split tensors along a new dimension
        t2 = torch.stack(t1, dim=2)  # Stack along a new dimension (dim=2)
        
        # Step 3: Apply the hyperbolic tangent function
        t3 = torch.tanh(t2)
        
        return t3

# Initializing the model with split sections
split_sections = 2  # Example: split the input tensor into 2 sections along the channel dimension
model = SplitTanhModel(split_sections)

# Inputs to the model
input_tensor = torch.randn(1, 4, 64, 64)  # Example input tensor of shape (batch_size, channels, height, width)
output = model(input_tensor)

print("Output shape:", output.shape)
