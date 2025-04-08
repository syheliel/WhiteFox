import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Using a 2D convolution layer
        self.conv_module = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # Batch normalization for 2D convolutions
        self.bn_module = torch.nn.BatchNorm2d(16, track_running_stats=True)

    def forward(self, x):
        conv_out = self.conv_module(x)  # Apply convolution to the input tensor
        bn_out = self.bn_module(conv_out)  # Apply batch normalization to the output of the convolution
        return bn_out

# Initializing the model
model = Model()

# Set the batch normalization module to evaluation mode
model.bn_module.eval()

# Example input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image
output = model(input_tensor)

print(output.shape)  # Outputs the shape of the result tensor
