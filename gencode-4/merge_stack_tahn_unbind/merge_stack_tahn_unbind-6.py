import torch

class Model(torch.nn.Module):
    def __init__(self, split_sections):
        super().__init__()
        self.split_sections = split_sections

    def forward(self, x):
        # Split the input tensor into chunks along the first dimension
        t1 = torch.split(x, self.split_sections, dim=0)
        # Stack the chunks along a new dimension (the second dimension)
        t2 = torch.stack(t1, dim=1)
        # Apply the hyperbolic tangent function to the stacked output
        t3 = torch.tanh(t2)
        return t3

# Initialize the model with a specified split size
split_sections = 2  # Example: split input tensor into 2 chunks
model = Model(split_sections)

# Create an example input tensor
# For example, a tensor of shape (4, 3, 64, 64) to be split into 2 chunks along the first dimension
input_tensor = torch.randn(4, 3, 64, 64)

# Pass the input tensor through the model
output = model(input_tensor)

# Output shape
print(output.shape)  # Should be (4, 2, 3, 64, 64)
