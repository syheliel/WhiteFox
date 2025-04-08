import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
    
    def forward(self, input_tensor, other):
        t1 = self.conv(input_tensor)  # Apply pointwise convolution
        t2 = t1 + other  # Add another tensor to the output of the convolution
        return t2

# Initializing the model
model = Model()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Example input tensor
other_tensor = torch.randn(1, 8, 64, 64)  # Another tensor to add to the output

# Getting the output from the model
output = model(input_tensor, other_tensor)

# Output shape for verification
print(output.shape)  # Should be [1, 8, 64, 64]
