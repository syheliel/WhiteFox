import torch

class Model(torch.nn.Module):
    def __init__(self, split_sections):
        super(Model, self).__init__()
        self.split_sections = split_sections

    def forward(self, x):
        # Step 1: Split the input tensor into chunks along the first dimension
        t1 = torch.split(x, self.split_sections, dim=1)  # Splitting along dimension 1
        # Step 2: Stack the sequence of tensors along a new dimension
        t2 = torch.stack(t1, dim=2)  # Stacking along a new dimension (dimension 2)
        # Step 3: Apply the hyperbolic tangent function to the output of the stack operation
        t3 = torch.tanh(t2)
        return t3

# Initializing the model with a specified split size
split_size = 2  # For instance, we want to split the first dimension into chunks of size 2
model = Model(split_sections=split_size)

# Generating an input tensor for the model
input_tensor = torch.randn(1, 4, 64, 64)  # Example: batch size 1, 4 channels, 64x64 spatial dimensions

# Forward pass through the model
output = model(input_tensor)

# Displaying the output shape
print("Output shape:", output.shape)
