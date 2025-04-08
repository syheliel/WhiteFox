import torch

# Define a new model
class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Apply pointwise convolution
        conv_output = self.conv(x)
        
        # Perform the required operations
        half_output = conv_output * 0.5
        scaled_output = conv_output * 0.7071067811865476
        erf_output = torch.erf(scaled_output)
        shifted_output = erf_output + 1
        final_output = half_output * shifted_output
        
        return final_output

# Instantiate the model
model = CustomModel()

# Generate an input tensor
input_tensor = torch.randn(1, 3, 128, 128)  # Batch size of 1, 3 channels, 128x128 image

# Forward pass through the model
output_tensor = model(input_tensor)

# Display the output tensor shape
print(output_tensor.shape)
