import torch

class CumulativeSumModel(torch.nn.Module):
    def __init__(self, arg1, arg2, dtype=torch.float32, layout=torch.strided, device='cpu'):
        super().__init__()
        self.arg1 = arg1
        self.arg2 = arg2
        self.dtype = dtype
        self.layout = layout
        self.device = device

    def forward(self):
        # Create a tensor filled with 1
        t1 = torch.full((self.arg1, self.arg2), 1, dtype=self.dtype, layout=self.layout, device=self.device, pin_memory=False)
        # Convert the tensor to the specified dtype
        t2 = t1.to(self.dtype)
        # Compute the cumulative sum along dimension 1
        t3 = torch.cumsum(t2, dim=1)
        return t3

# Initializing the model
model = CumulativeSumModel(arg1=4, arg2=5, dtype=torch.float32)

# Generate the output tensor from the model
output_tensor = model()
print(output_tensor)
