import torch

class SigmoidModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Applying a pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)
 
    def forward(self, input_tensor):
        t1 = self.conv(input_tensor)  # Apply pointwise convolution
        t2 = torch.sigmoid(t1)        # Apply the sigmoid function
        return t2

# Initializing the model
model = SigmoidModel()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 input channels, 64x64 image size
output = model(input_tensor)

# Display the output shape
print("Output shape:", output.shape)
