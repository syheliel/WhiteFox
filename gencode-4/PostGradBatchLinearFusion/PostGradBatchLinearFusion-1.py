import torch

# Define the model
class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.pointwise_conv = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor):
        conv_output = self.pointwise_conv(input_tensor)  # Apply pointwise convolution
        scaled_output_half = conv_output * 0.5  # Multiply by 0.5
        scaled_output_sqrt2 = conv_output * 0.7071067811865476  # Multiply by 0.7071067811865476
        error_function_output = torch.erf(scaled_output_sqrt2)  # Apply the error function
        shifted_output = error_function_output + 1  # Add 1
        final_output = scaled_output_half * shifted_output  # Final multiplication
        return final_output

# Initialize the model
model = CustomModel()

# Generate an input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, height and width of 64

# Get the output from the model
output = model(input_tensor)

print(output.shape)  # Display the shape of the output tensor
