import torch

# Model definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, 1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv_transpose(x)          # Apply pointwise transposed convolution
        t2 = torch.sigmoid(t1)               # Apply the sigmoid function
        t3 = t1 * t2                         # Multiply the output of transposed convolution by the sigmoid output
        return t3

# Initializing the model
model = Model()

# Generating input tensor
input_tensor = torch.randn(1, 8, 64, 64)  # Assuming 8 input channels and spatial dimensions 64x64

# Running the model with the input tensor
output_tensor = model(input_tensor)

# Printing the shape of the output tensor
print(output_tensor.shape)
