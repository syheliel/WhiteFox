import torch

class PermuteLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a linear layer with input features and output features
        self.linear = torch.nn.Linear(32, 16)

    def forward(self, input_tensor):
        # Permute the input tensor to swap the last two dimensions
        t1 = input_tensor.permute(0, 2, 1, 3)  # Assuming input_tensor is of shape (N, C, H, W)
        # Reshape t1 before passing it to the linear layer
        N, C, H, W = t1.shape
        t1 = t1.contiguous().view(N * H * W, C)  # Reshape to (N * H * W, C)
        # Apply the linear transformation
        t2 = self.linear(t1)
        # Reshape back to (N, H, W, output_features)
        t2 = t2.view(N, H, W, -1)
        return t2

# Initializing the model
model = PermuteLinearModel()

# Inputs to the model
input_tensor = torch.randn(1, 3, 8, 8)  # Example input tensor with shape (N, C, H, W)
output = model(input_tensor)

# Output shape
print(output.shape)  # This will show the shape of the output tensor
