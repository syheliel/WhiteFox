import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input_tensor):
        # Split the input tensor into chunks of size 16 along dimension 1
        split_sections = 16
        t1 = torch.split(input_tensor, split_sections, dim=1) 
        
        # Select chunks at indices 0, 1, and 2
        indices = [0, 1, 2]
        t2 = [t1[i] for i in indices] 
        
        # Concatenate the selected chunks along dimension 1
        t3 = torch.cat(t2, dim=1) 
        
        return t3

# Initializing the model
model = Model()

# Inputs to the model
input_tensor = torch.randn(1, 48, 64, 64)  # 48 channels, so we can split into 3 chunks of 16
output = model(input_tensor)

print("Output shape:", output.shape)
