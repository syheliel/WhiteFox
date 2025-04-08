import torch

# Define the model class
class AddMMModel(torch.nn.Module):
    def __init__(self):
        super(AddMMModel, self).__init__()
        # Create parameters for the matrix multiplication
        self.mat1 = torch.randn(10240, 32, device='cuda', requires_grad=True)
        self.mat2 = torch.randn(32, 32, device='cuda', requires_grad=True)

    def forward(self, input):
        # Perform matrix multiplication and addition
        output = torch.addmm(input, self.mat1, self.mat2)
        return output

# Initialize the model
model = AddMMModel()

# Inputs to the model
# Create a random input tensor of shape (10240, 32)
input_tensor = torch.randn(10240, 32, device='cuda')

# Forward pass through the model
output = model(input_tensor)

# Print the shape of the output
print(output.shape)  # This should output: torch.Size([10240, 32])
