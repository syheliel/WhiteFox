import torch

# Model Definition
class ModelTransposeConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x1):
        t1 = self.conv_transpose(x1)  # Apply pointwise transposed convolution
        t2 = t1 + 3                    # Add 3 to the output of the transposed convolution
        t3 = torch.clamp_min(t2, 0)    # Clamp the output of the addition operation to a minimum of 0
        t4 = torch.clamp_max(t3, 6)    # Clamp the output of the previous operation to a maximum of 6
        t5 = t4 / 6                     # Divide the output of the previous operation by 6
        return t5

# Initializing the model
model = ModelTransposeConv()

# Inputs to the model
input_tensor = torch.randn(1, 8, 64, 64)  # Input tensor with shape [batch_size, channels, height, width]
output = model(input_tensor)

# Display output shape
print(output.shape)  # The output shape will be [1, 3, 64, 64]
