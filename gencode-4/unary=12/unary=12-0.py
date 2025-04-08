import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x1):
        t1 = self.conv(x1)  # Apply pointwise convolution
        t2 = torch.sigmoid(t1)  # Apply the sigmoid function
        t3 = t1 * t2  # Multiply the convolution output by the sigmoid output
        return t3

# Initializing the model
model = Model()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 input channels, 64x64 spatial dimensions
output_tensor = model(input_tensor)

print("Input Tensor Shape:", input_tensor.shape)
print("Output Tensor Shape:", output_tensor.shape)
