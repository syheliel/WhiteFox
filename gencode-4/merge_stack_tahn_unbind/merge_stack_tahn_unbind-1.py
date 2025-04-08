import torch

class SplitAndTanhModel(torch.nn.Module):
    def __init__(self, split_sections):
        super().__init__()
        self.split_sections = split_sections

    def forward(self, x):
        t1 = torch.split(x, self.split_sections, dim=1)  # Split the input tensor
        t2 = torch.stack(t1, dim=1)  # Stack the split tensors along a new dimension
        t3 = torch.tanh(t2)  # Apply the hyperbolic tangent function
        return t3

# Initialize the model with specific split sections
split_sections = 2  # This would mean splitting the input tensor along dimension 1 into chunks of 2
model = SplitAndTanhModel(split_sections)

# Input tensor for the model
input_tensor = torch.randn(1, 4, 64, 64)  # A random tensor with shape (batch_size=1, channels=4, height=64, width=64)

# Get the output from the model
output = model(input_tensor)

# Print the output shape
print("Output shape:", output.shape)
