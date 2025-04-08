import torch
import torch.nn as nn

class SelfAttentionModel(nn.Module):
    def __init__(self, embed_size, heads, dropout_p):
        super(SelfAttentionModel, self).__init__()
        self.heads = heads
        self.embed_size = embed_size
        self.dropout_p = dropout_p
        self.scale = embed_size ** 0.5
        
        # Define linear transformations for query, key, and value
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, attn_mask):
        bs, seq_length, _ = x.size()
        
        q = self.query(x).view(bs, seq_length, self.heads, -1).permute(0, 2, 1, 3)  # (bs, heads, seq_length, head_dim)
        k = self.key(x).view(bs, seq_length, self.heads, -1).permute(0, 2, 1, 3)  # (bs, heads, seq_length, head_dim)
        v = self.value(x).view(bs, seq_length, self.heads, -1).permute(0, 2, 1, 3)  # (bs, heads, seq_length, head_dim)

        k_len = k.size(-2)  # Length of the key sequence
        scores = q @ k.transpose(-2, -1)  # Dot product (bs, heads, seq_length, seq_length)
        scores = scores / self.scale  # Scale scores

        fill_value = torch.full((), -float("inf"), dtype=x.dtype, device=x.device)  # Tensor filled with -inf
        attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)  # Expand attention mask
        attention_weights = torch.nn.functional.dropout(
            torch.softmax(scores.masked_fill(attn_mask, fill_value), dim=-1),
            self.dropout_p
        )

        output = attention_weights @ v  # (bs, heads, seq_length, head_dim)
        return output

# Initialize the model
embed_size = 64  # Size of the embedding
heads = 8       # Number of attention heads
dropout_p = 0.1 # Dropout probability
model = SelfAttentionModel(embed_size=embed_size, heads=heads, dropout_p=dropout_p)

# Create input tensor and attention mask
seq_length = 10  # Length of the input sequence
x_input = torch.randn(1, seq_length, embed_size)  # Input tensor (batch_size, seq_length, embed_size)
attn_mask = torch.randint(0, 2, (1, seq_length))  # Attention mask (1 for valid, 0 for masked)

# Forward pass through the model
output = model(x_input, attn_mask)

print("Output shape:", output.shape)  # Should be (1, heads, seq_length, head_dim)
