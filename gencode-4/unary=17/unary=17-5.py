import torch

# Model definition
class TransposedConvReLUModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a pointwise transposed convolution layer
        self.conv_transpose = torch.nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        t1 = self.conv_transpose(x)  # Apply pointwise transposed convolution
        t2 = torch.nn.functional.relu(t1)  # Apply ReLU activation function
        return t2

# Initializing the model
model = TransposedConvReLUModel()

# Inputs to the model
input_tensor = torch.randn(1, 8, 32, 32)  # Example input tensor with batch size 1, 8 channels, and 32x32 spatial dimensions
output = model(input_tensor)

# Output shape
print("Output shape:", output.shape)
