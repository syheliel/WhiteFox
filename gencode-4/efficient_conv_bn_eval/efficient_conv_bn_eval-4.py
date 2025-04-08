import torch

# Define the model
class ConvolutionalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Convolution layer with 3 input channels, 16 output channels, kernel size of 3
        self.conv_module = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        # Batch normalization layer for 16 channels
        self.bn_module = torch.nn.BatchNorm2d(num_features=16, track_running_stats=True)

    def forward(self, x):
        # Apply convolution to the input tensor
        conv_out = self.conv_module(x)
        # Apply batch normalization to the output of the convolution
        bn_out = self.bn_module(conv_out)
        return bn_out

# Initialize the model
model = ConvolutionalModel()

# Set batch normalization to evaluation mode
model.bn_module.eval()

# Generate input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image size

# Get the output of the model
output = model(input_tensor)

print(output.shape)  # To verify the output shape
