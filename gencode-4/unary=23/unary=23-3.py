import torch

# Define the model
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Pointwise transposed convolution layer
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv_transpose(x)  # Apply pointwise transposed convolution
        t2 = torch.tanh(t1)           # Apply hyperbolic tangent function
        return t2

# Initializing the model
model = Model()

# Generate input tensor for the model
input_tensor = torch.randn(1, 8, 64, 64)  # Batch size of 1, 8 input channels, spatial size 64x64

# Forward pass through the model
output = model(input_tensor)

# Display output shape
print(output.shape)
