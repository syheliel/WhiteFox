import torch

# Define the model
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv_module = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn_module = torch.nn.BatchNorm2d(num_features=16, track_running_stats=True)
    
    def forward(self, input_tensor):
        conv_out = self.conv_module(input_tensor)  # Apply convolution to the input tensor
        self.bn_module.eval()  # Set batch normalization to evaluation mode
        bn_out = self.bn_module(conv_out)  # Apply batch normalization to the output of the convolution
        return bn_out

# Initialize the model
model = MyModel()

# Generate an input tensor for the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 spatial dimensions

# Get the model's output
output = model(input_tensor)

# Print the output shape
print(output.shape)  # Should print a shape of (1, 16, 64, 64)
