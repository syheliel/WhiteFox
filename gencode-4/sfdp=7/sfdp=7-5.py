import torch
import math

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query = torch.nn.Linear(embed_dim, embed_dim)
        self.key = torch.nn.Linear(embed_dim, embed_dim)
        self.value = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # Assume x has shape (batch_size, seq_length, embed_dim)
        q = self.query(x).view(x.size(0), x.size(1), self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)
        k = self.key(x).view(x.size(0), x.size(1), self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)
        v = self.value(x).view(x.size(0), x.size(1), self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)

        div = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        div = div.to(torch.float32)
        attn_weight = torch.softmax(div, dim=-1)
        attn_weight = attn_weight.to(torch.float16)
        output = attn_weight @ v
        
        return output.permute(0, 2, 1, 3).contiguous().view(x.size(0), x.size(1), self.embed_dim)

# Initializing the model
embed_dim = 16  # Embedding dimension
num_heads = 4   # Number of attention heads
model = AttentionModel(embed_dim, num_heads)

# Example input tensor (batch_size=1, seq_length=10, embed_dim=16)
input_tensor = torch.randn(1, 10, embed_dim)

# Forward pass
output_tensor = model(input_tensor)

print("Output tensor shape:", output_tensor.shape)
