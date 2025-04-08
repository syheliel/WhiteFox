import torch

# Define the model
class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_module = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn_module = torch.nn.BatchNorm2d(num_features=16, track_running_stats=True)

    def forward(self, input_tensor):
        conv_out = self.conv_module(input_tensor)  # Apply convolution
        bn_out = self.bn_module(conv_out)          # Apply batch normalization
        return bn_out

# Initializing the model
model = CustomModel()

# Set the batch normalization module to evaluation mode
model.bn_module.eval()

# Generate a random input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image

# Get the output of the model
output = model(input_tensor)

# Output the shape of the result
print(output.shape)  # Expected shape will be (1, 16, 64, 64)
