import torch

# Model
class Model(torch.nn.Module):
    def __init__(self, min_value=0.0, max_value=1.0):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x1):
        t1 = self.conv_transpose(x1)  # Apply pointwise transposed convolution
        t2 = torch.clamp_min(t1, self.min_value)  # Clamp to minimum value
        t3 = torch.clamp_max(t2, self.max_value)  # Clamp to maximum value
        return t3

# Initializing the model
m = Model(min_value=0.0, max_value=1.0)

# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)  # Random input tensor with shape (batch_size, channels, height, width)
output = m(x1)

# Display the shape of the output
print(output.shape)
