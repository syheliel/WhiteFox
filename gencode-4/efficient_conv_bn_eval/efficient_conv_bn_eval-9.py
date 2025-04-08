import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a 2D convolutional layer
        self.conv_module = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        # Define a batch normalization layer
        self.bn_module = torch.nn.BatchNorm2d(num_features=16, track_running_stats=True)

    def forward(self, x):
        # Apply convolution
        conv_out = self.conv_module(x)
        # Apply batch normalization
        bn_out = self.bn_module(conv_out)
        return bn_out

# Initializing the model
model = Model()

# Set the batch normalization layer to evaluation mode
model.bn_module.eval()

# Generate input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image

# Forward pass through the model
output = model(input_tensor)

# Displaying the shape of the output tensor
print(output.shape)
