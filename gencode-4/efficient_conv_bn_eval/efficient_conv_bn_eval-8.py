import torch

# Model Definition
class SimpleConvBNModel(torch.nn.Module):
    def __init__(self):
        super(SimpleConvBNModel, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(num_features=16, track_running_stats=True)

    def forward(self, x):
        conv_out = self.conv(x)  # Apply convolution to the input tensor
        bn_out = self.bn(conv_out)  # Apply batch normalization to the output of the convolution
        return bn_out

# Initializing the model
model = SimpleConvBNModel()

# Set the batch normalization module to evaluation mode
model.bn.training = False

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image
output = model(input_tensor)

# Display output shape
print(output.shape)
