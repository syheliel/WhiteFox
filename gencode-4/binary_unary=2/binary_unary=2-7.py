import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)  # Pointwise convolution

    def forward(self, input_tensor):
        t1 = self.conv(input_tensor)  # Apply pointwise convolution
        other = 0.5  # Scalar to subtract
        t2 = t1 - other  # Subtract scalar from the output of the convolution
        t3 = torch.relu(t2)  # Apply ReLU activation function
        return t3

# Initializing the model
model = Model()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Example input tensor
output = model(input_tensor)  # Get the output from the model
