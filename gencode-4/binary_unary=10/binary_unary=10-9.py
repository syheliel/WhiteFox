import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 input features to 5 output features
        self.other = torch.randn(1, 5)  # Another tensor to be added, of the same size as the output of the linear layer

    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = t1 + self.other  # Add another tensor
        t3 = torch.relu(t2)   # Apply ReLU activation
        return t3

# Initializing the model
model = MyModel()

# Input to the model
input_tensor = torch.randn(1, 10)  # A batch of size 1 with 10 features
output = model(input_tensor)

print("Input Tensor:", input_tensor)
print("Output Tensor:", output)
