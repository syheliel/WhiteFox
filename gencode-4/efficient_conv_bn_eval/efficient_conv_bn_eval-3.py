import torch

# Model definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_module = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn_module = torch.nn.BatchNorm2d(num_features=16, track_running_stats=True)
        
    def forward(self, x):
        conv_out = self.conv_module(x)  # Apply convolution
        self.bn_module.training = False  # Set to evaluation mode
        bn_out = self.bn_module(conv_out)  # Apply batch normalization
        return bn_out

# Initializing the model
model = Model()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image
output = model(input_tensor)

print("Output shape:", output.shape)  # Should print the shape of the output tensor
