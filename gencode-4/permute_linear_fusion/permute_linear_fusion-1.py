import torch

# Define the model
class PermuteLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a linear layer with an appropriate input size after permutation
        self.linear = torch.nn.Linear(64 * 32, 128)  # Example: input of size 64*32, output of size 128

    def forward(self, input_tensor):
        # Permute the tensor to swap the last two dimensions
        t1 = input_tensor.permute(0, 2, 1, 3)  # Example permutation for a tensor of shape (N, C, H, W)
        t1 = t1.contiguous()  # Ensure that the tensor is contiguous after permutation
        t1 = t1.view(t1.size(0), -1, t1.size(2))  # Flatten the last two dimensions for linear layer
        t2 = self.linear(t1)  # Apply linear transformation
        return t2

# Initialize the model
model = PermuteLinearModel()

# Create an input tensor with shape (N, C, H, W)
input_tensor = torch.randn(1, 32, 64, 64)  # Batch size of 1, 32 channels, 64x64 spatial dimensions

# Forward pass
output = model(input_tensor)

# Display the output shape
print(output.shape)  # Should be (1, 128, 64)
