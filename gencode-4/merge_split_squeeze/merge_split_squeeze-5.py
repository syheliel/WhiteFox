import torch

# Define the model
class SplitSqueezeModel(torch.nn.Module):
    def __init__(self):
        super(SplitSqueezeModel, self).__init__()

    def forward(self, input_tensor):
        # Split the input tensor into chunks of size 1 along dimension 1
        split_sizes = [1] * input_tensor.size(1) # Create a list of integers where all elements are 1
        split_tensor = torch.split(input_tensor, split_sizes, dim=1)
        
        # Squeeze each chunk of the split tensor along dimension 1
        squeezed_tensors = [torch.squeeze(t, dim=1) for t in split_tensor]
        
        return squeezed_tensors

# Initialize the model
model = SplitSqueezeModel()

# Generate an input tensor
# For this example, let's create a tensor of shape (1, 5, 64, 64)
input_tensor = torch.randn(1, 5, 64, 64)

# Run the model with the input tensor
output_tensors = model(input_tensor)

# Output the shapes of the squeezed tensors
for i, out in enumerate(output_tensors):
    print(f"Squeezed tensor {i} shape: {out.shape}")
