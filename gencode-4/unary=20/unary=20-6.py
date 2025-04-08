import torch

# Define the model
class ModelTransposeConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution (1x1 kernel)
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv_transpose(x)  # Apply pointwise transposed convolution
        t2 = torch.sigmoid(t1)        # Apply the sigmoid function
        return t2

# Initializing the model
model = ModelTransposeConv()

# Generate input tensor
# Let's assume we are using a batch size of 1 and 8 channels with height and width of 64
input_tensor = torch.randn(1, 8, 64, 64)

# Forward pass through the model
output = model(input_tensor)

# Output shape
print("Output shape:", output.shape)
