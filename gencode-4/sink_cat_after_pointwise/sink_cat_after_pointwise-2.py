import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, tensor1, tensor2):
        # Concatenate tensors along the channel dimension
        t1 = torch.cat([tensor1, tensor2], dim=1)
        # Reshape the concatenated tensor
        t2 = t1.view(t1.size(0), -1)  # Flatten the tensor while maintaining batch size
        # Apply ReLU activation
        t3 = torch.relu(t2)
        return t3

# Initializing the model
m = Model()

# Inputs to the model
tensor1 = torch.randn(1, 3, 64, 64)  # A random tensor with shape (batch_size, channels, height, width)
tensor2 = torch.randn(1, 3, 64, 64)  # Another random tensor with the same shape
output = m(tensor1, tensor2)

print(output.shape)  # Output shape after processing through the model
