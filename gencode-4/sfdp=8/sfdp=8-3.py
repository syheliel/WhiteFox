import torch
import torch.nn.functional as F
import math

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        
        # Define layers for query, key, and value projections
        self.query_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.key_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.value_proj = torch.nn.Linear(embed_dim, embed_dim)
        
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, query, key, value):
        # Permute the dimensions of the query, key, and value tensors
        q = self.query_proj(query).permute(0, 2, 1, 3)  # (batch_size, seq_len, heads, head_dim)
        k = self.key_proj(key).permute(0, 2, 1, 3)      # (batch_size, seq_len, heads, head_dim)
        v = self.value_proj(value).permute(0, 2, 1, 3)  # (batch_size, seq_len, heads, head_dim)

        # Scale the query tensor by the square root of its last dimension size
        q = q / math.sqrt(q.size(-1))

        # Perform matrix multiplication between query and transposed key
        div = q @ k.transpose(-2, -1)  # (batch_size, heads, seq_len_q, seq_len_k)

        # Convert the result to float32
        div = div.to(torch.float32)

        # Apply softmax to the result along the last dimension
        attn_weight = F.softmax(div, dim=-1)

        # Apply dropout to the softmax result
        attn_weight = self.dropout(attn_weight)

        # Convert the result to float16
        attn_weight = attn_weight.to(torch.float16)

        # Perform matrix multiplication between the result and the value tensor
        output = attn_weight @ v  # (batch_size, heads, seq_len_q, head_dim)

        return output

# Initializing the model
embed_dim = 64
num_heads = 8
dropout_p = 0.1
model = AttentionModel(embed_dim, num_heads, dropout_p)

# Inputs to the model
batch_size = 1
seq_len = 10
x_query = torch.randn(batch_size, seq_len, embed_dim)
x_key = torch.randn(batch_size, seq_len, embed_dim)
x_value = torch.randn(batch_size, seq_len, embed_dim)

# Forward pass
output = model(x_query, x_key, x_value)

# Output tensor shape
print(output.shape)  # Should be (batch_size, num_heads, seq_len_q, head_dim)
