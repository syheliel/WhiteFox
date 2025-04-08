import torch

class PermuteLinearModel(torch.nn.Module):
    def __init__(self):
        super(PermuteLinearModel, self).__init__()
        # Defining a linear layer with input features matching the permuted dimensions
        self.linear = torch.nn.Linear(64, 32)

    def forward(self, x):
        # Permute the input tensor (assumes x has shape [batch_size, channels, height, width])
        # This will swap the last two dimensions (height and width)
        t1 = x.permute(0, 1, 3, 2)  # Change from [N, C, H, W] to [N, C, W, H]
        
        # Flatten the permuted tensor from [N, C, W, H] to [N * W, C] for linear layer
        t1_flattened = t1.view(-1, t1.size(1))

        # Apply linear transformation
        t2 = self.linear(t1_flattened)

        # Reshape back to [N, W, output_features] if needed, here it will depend on your use case
        return t2.view(x.size(0), t1.size(2), -1)  # [N, W, output_features]

# Initializing the model
model = PermuteLinearModel()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Example input with shape [batch_size, channels, height, width]

# Get the output from the model
output = model(input_tensor)

print("Output shape:", output.shape)
