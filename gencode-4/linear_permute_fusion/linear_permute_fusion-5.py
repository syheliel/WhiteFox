import torch

class LinearPermuteModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define weights and bias for the linear layer
        self.weight = torch.nn.Parameter(torch.randn(8, 4))  # Assuming input features of size 4 and output features of size 8
        self.bias = torch.nn.Parameter(torch.randn(8))

    def forward(self, x):
        # Apply linear transformation
        t1 = torch.nn.functional.linear(x, self.weight, self.bias)
        # Permute the output tensor (assuming t1 has shape [batch_size, channels, height, width])
        # For example, let's say t1 has shape (N, C, H, W), we will swap H and W: (N, C, W, H)
        t2 = t1.permute(0, 2, 1)  # Swap last two dimensions (only works if t1 has more than 2 dimensions)
        return t2

# Initializing the model
model = LinearPermuteModel()

# Inputs to the model (shape: [batch_size, input_features])
# Here we assume input size is (1, 4) to match the linear layer's input
input_tensor = torch.randn(1, 4)

# Get the output from the model
output = model(input_tensor)

print("Output shape:", output.shape)
