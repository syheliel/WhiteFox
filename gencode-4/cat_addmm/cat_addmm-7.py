import torch

# Model Definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mat1 = torch.nn.Parameter(torch.randn(10, 5))  # Weight matrix 1
        self.mat2 = torch.nn.Parameter(torch.randn(5, 10))  # Weight matrix 2

    def forward(self, input_tensor):
        # Perform a matrix multiplication of mat1 and mat2 and add it to the input
        t1 = torch.addmm(input_tensor, self.mat1, self.mat2)
        # Concatenate the result along a specified dimension (dim=1)
        t2 = torch.cat([t1], dim=1)
        return t2

# Initializing the model
model = Model()

# Inputs to the model
input_tensor = torch.randn(1, 10)  # Batch size of 1 and input size of 10
output = model(input_tensor)

# Displaying the output shape
print(output.shape)
