import torch
import torch.nn as nn

class SelfAttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attn_mask, inv_scale):
        q = self.query(x).view(x.size(0), -1, self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        k = self.key(x).view(x.size(0), -1, self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        v = self.value(x).view(x.size(0), -1, self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]

        t1 = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, num_heads, seq_len, seq_len]
        t2 = t1.div(inv_scale)  # [batch_size, num_heads, seq_len, seq_len]
        t3 = t2 + attn_mask  # [batch_size, num_heads, seq_len, seq_len]
        t4 = t3.softmax(dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        t5 = t4.matmul(v)  # [batch_size, num_heads, seq_len, head_dim]
        
        return t5

# Initialize the model
embed_dim = 64  # Example embedding dimension
num_heads = 8   # Example number of heads
model = SelfAttentionModel(embed_dim, num_heads)

# Generate an input tensor
batch_size = 1
seq_length = 10  # Example sequence length
attn_mask = torch.zeros(batch_size, num_heads, seq_length, seq_length)  # Example attention mask
inv_scale = 1.0 / (embed_dim ** 0.5)  # Inverse scale for attention

x = torch.randn(batch_size, seq_length, embed_dim)  # Input tensor: [batch_size, seq_length, embed_dim]
output = model(x, attn_mask, inv_scale)

print("Output shape:", output.shape)  # Output from the model
