import torch
import math

class SelfAttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.query = torch.nn.Linear(embed_dim, embed_dim)
        self.key = torch.nn.Linear(embed_dim, embed_dim)
        self.value = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # Assuming x has shape (batch_size, seq_length, embed_dim)
        q = self.query(x).permute(0, 2, 1)  # Shape: (batch_size, embed_dim, seq_length)
        k = self.key(x).permute(0, 2, 1)    # Shape: (batch_size, embed_dim, seq_length)
        v = self.value(x).permute(0, 2, 1)  # Shape: (batch_size, embed_dim, seq_length)

        q = q / math.sqrt(q.size(-1))       # Scale the query tensor
        div = q @ k.transpose(-2, -1)       # Matrix multiplication
        div = div.to(torch.float32)          # Convert to float32

        attn_weight = torch.softmax(div, dim=-1)  # Apply softmax
        attn_weight = attn_weight.to(torch.float16)  # Convert to float16

        output = attn_weight @ v  # Final matrix multiplication
        return output

# Initialize the model with example parameters
embed_dim = 64  # Example embedding dimension
num_heads = 8   # Example number of heads in multi-head attention
model = SelfAttentionModel(embed_dim, num_heads)

# Create an input tensor with shape (batch_size, seq_length, embed_dim)
batch_size = 1
seq_length = 10  # Example sequence length
x_input = torch.randn(batch_size, seq_length, embed_dim)

# Get the output of the model
output = model(x_input)
print(output.shape)  # This will print the shape of the output tensor
