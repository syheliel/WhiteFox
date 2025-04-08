import torch

# Define the model
class SimpleModel(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(input_features, output_features)

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation to the input tensor
        t2 = torch.sigmoid(t1)  # Apply the sigmoid function to the output of the linear transformation
        return t2

# Initialize the model with input and output feature sizes
input_features = 10  # Number of input features
output_features = 1  # Number of output features
model = SimpleModel(input_features, output_features)

# Generate input tensor
# Create a random input tensor with shape (batch_size, input_features)
batch_size = 5  # Number of samples in a batch
input_tensor = torch.randn(batch_size, input_features)

# Forward pass through the model
output = model(input_tensor)

# Print the output
print("Input Tensor:")
print(input_tensor)
print("\nOutput Tensor:")
print(output)
