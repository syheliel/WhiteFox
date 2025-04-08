import torch

class TransposeConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution with kernel size 1
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 4, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv_transpose(x)  # Apply pointwise transposed convolution
        t2 = torch.nn.functional.relu(t1)  # Apply ReLU activation
        return t2

# Initializing the model
model = TransposeConvModel()

# Inputs to the model
input_tensor = torch.randn(1, 16, 32, 32)  # Batch size of 1, 16 channels, height and width of 32
output_tensor = model(input_tensor)

# Output for verification
print(output_tensor.shape)  # Should be (1, 4, 32, 32) after transposed convolution
