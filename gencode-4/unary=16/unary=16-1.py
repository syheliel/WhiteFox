import torch

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(128, 64)  # Linear transformation from 128 input features to 64 output features

    def forward(self, x):
        t1 = self.linear(x)  # Apply the linear transformation
        t2 = torch.relu(t1)  # Apply the ReLU activation function
        return t2

# Initializing the model
model = SimpleModel()

# Inputs to the model
input_tensor = torch.randn(1, 128)  # Batch size of 1 and 128 features
output = model(input_tensor)

print("Output shape:", output.shape)
