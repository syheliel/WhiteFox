import torch
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(num_features=8)

        # Setting the running mean and variance to constant values for demonstration
        self.bn.running_mean = torch.tensor([0.0] * 8)
        self.bn.running_var = torch.tensor([1.0] * 8)
        self.bn.weight = torch.tensor([1.0] * 8, requires_grad=False)
        self.bn.bias = torch.tensor([0.0] * 8, requires_grad=False)

    def forward(self, x):
        self.bn.eval()  # Set the batch norm layer to evaluation mode
        conv_output = self.conv(x)
        output = self.bn(conv_output)
        return output

# Initializing the model
model = Model()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image
output = model(input_tensor)

print("Output shape:", output.shape)
