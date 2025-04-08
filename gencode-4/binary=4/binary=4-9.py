import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear layer from 10 input features to 5 output features
        self.other = torch.randn(1, 5)  # Another tensor with the same shape as the output of the linear layer

    def forward(self, input_tensor):
        t1 = self.linear(input_tensor)  # Apply linear transformation
        t2 = t1 + self.other  # Add another tensor to the output
        return t2

# Initializing the model
model = Model()

# Generating input tensor
input_tensor = torch.randn(1, 10)  # Batch size of 1 and 10 input features

# Forward pass
output = model(input_tensor)
