import torch

# Model Definition
class Model(torch.nn.Module):
    def __init__(self, min_value=0.0, max_value=1.0):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, 1, stride=1, padding=0)
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x1):
        t1 = self.conv_transpose(x1)  # Apply pointwise transposed convolution
        t2 = torch.clamp_min(t1, self.min_value)  # Clamp to a minimum value
        t3 = torch.clamp_max(t2, self.max_value)  # Clamp to a maximum value
        return t3

# Initializing the model
model = Model(min_value=0.0, max_value=1.0)

# Inputs to the model
# Creating an input tensor with shape (batch_size=1, channels=8, height=64, width=64)
input_tensor = torch.randn(1, 8, 64, 64)

# Forward pass through the model
output_tensor = model(input_tensor)

# Output tensor shape
print(output_tensor.shape)
