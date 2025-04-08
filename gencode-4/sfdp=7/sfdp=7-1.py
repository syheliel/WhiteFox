import torch
import math

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionModel, self).__init__()
        self.query = torch.nn.Linear(embed_dim, embed_dim)
        self.key = torch.nn.Linear(embed_dim, embed_dim)
        self.value = torch.nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward(self, x):
        # Assume x has shape (batch_size, seq_length, embed_dim)
        q = self.query(x).view(x.size(0), -1, self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)
        k = self.key(x).view(x.size(0), -1, self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)
        v = self.value(x).view(x.size(0), -1, self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)

        # Compute scaled dot-product attention
        div = (q @ k.transpose(-2, -1)) / math.sqrt(self.embed_dim // self.num_heads)
        div = div.to(torch.float32)
        attn_weight = torch.softmax(div, dim=-1)
        attn_weight = attn_weight.to(torch.float16)

        output = attn_weight @ v
        return output.view(x.size(0), -1, self.embed_dim)  # Reshape back to (batch_size, seq_length, embed_dim)

# Initializing the model
embed_dim = 64  # Example embedding dimension
num_heads = 8   # Example number of attention heads
model = AttentionModel(embed_dim, num_heads)

# Inputs to the model
batch_size = 2    # Example batch size
seq_length = 10   # Example sequence length
x = torch.randn(batch_size, seq_length, embed_dim)

# Forward pass
output = model(x)
