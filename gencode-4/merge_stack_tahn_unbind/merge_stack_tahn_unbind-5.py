import torch

class Model(torch.nn.Module):
    def __init__(self, split_sections):
        super().__init__()
        self.split_sections = split_sections

    def forward(self, x1):
        # Step 1: Split the input tensor into chunks along the specified dimension
        t1 = torch.split(x1, self.split_sections, dim=1)  # Split along dimension 1
        
        # Step 2: Stack the split tensors along a new dimension
        t2 = torch.stack(t1, dim=2)  # Stack along a new dimension (dimension 2)
        
        # Step 3: Apply the hyperbolic tangent function
        t3 = torch.tanh(t2)
        
        return t3

# Initializing the model with a specified split size
split_size = 1  # Split the input tensor along dim=1 into chunks of size 1
model = Model(split_sections=split_size)

# Inputs to the model
input_tensor = torch.randn(1, 4, 64, 64)  # Batch size of 1, 4 channels, 64x64 image
output_tensor = model(input_tensor)

print(output_tensor.shape)
