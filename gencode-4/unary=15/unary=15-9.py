import torch

class SimpleReLUConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution (kernel size 1)
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor):
        t1 = self.conv(input_tensor)  # Apply pointwise convolution
        t2 = torch.relu(t1)           # Apply ReLU activation
        return t2

# Initializing the model
model = SimpleReLUConvModel()

# Generating input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image

# Forward pass through the model
output = model(input_tensor)

print("Output shape:", output.shape)
