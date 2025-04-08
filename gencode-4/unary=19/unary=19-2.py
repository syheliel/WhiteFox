import torch

# Model Definition
class SigmoidModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        t1 = self.linear(x)              # Apply a linear transformation to the input tensor
        t2 = torch.sigmoid(t1)           # Apply the sigmoid function to the output of the linear transformation
        return t2

# Initializing the model
input_size = 10  # Example input size
output_size = 1  # Example output size
model = SigmoidModel(input_size, output_size)

# Creating an input tensor
input_tensor = torch.randn(1, input_size)  # Batch size of 1

# Perform a forward pass
output = model(input_tensor)

print("Input Tensor:", input_tensor)
print("Output Tensor:", output)
