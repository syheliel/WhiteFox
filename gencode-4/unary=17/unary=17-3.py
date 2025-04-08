import torch

# Model definition
class TransposeConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution with kernel size 1
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, 1, stride=1, padding=0)
        
    def forward(self, x1):
        t1 = self.conv_transpose(x1)  # Apply pointwise transposed convolution
        t2 = torch.relu(t1)            # Apply the ReLU activation function
        return t2

# Initializing the model
model = TransposeConvModel()

# Generating an input tensor for the model
# Assuming we want an input tensor with 8 channels and a spatial dimension of 64x64
input_tensor = torch.randn(1, 8, 64, 64)  # Batch size of 1, 8 channels, 64x64 spatial dimensions

# Getting the output from the model
output = model(input_tensor)

# Checking the output shape
print("Output shape:", output.shape)
