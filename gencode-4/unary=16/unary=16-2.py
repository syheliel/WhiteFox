import torch

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # Define a linear layer with an input size of 128 and an output size of 64
        self.linear = torch.nn.Linear(128, 64)

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation to the input tensor
        t2 = torch.relu(t1)  # Apply the ReLU activation function to the output of the linear transformation
        return t2

# Initializing the model
model = SimpleModel()

# Inputs to the model
input_tensor = torch.randn(1, 128)  # Create a random input tensor with batch size 1 and 128 features
output_tensor = model(input_tensor)  # Forward pass through the model

print("Input Tensor Shape:", input_tensor.shape)
print("Output Tensor Shape:", output_tensor.shape)
