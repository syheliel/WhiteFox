import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(AttentionModel, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.scale = embed_size ** 0.5  # Inverse scale for attention
        
        # Linear layers for query, key, and value projections
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, attn_mask):
        # Permute the dimensions of the query, key, and value tensors
        q = self.query_linear(query).permute(0, 2, 1, 3)
        k = self.key_linear(key).permute(0, 2, 1, 3)
        v = self.value_linear(value).permute(0, 2, 1, 3)

        # Perform matrix multiplication between the query tensor and the transposed key tensor
        t1 = torch.matmul(q, k.transpose(-2, -1))  # Shape: (batch_size, num_heads, seq_length, seq_length)
        
        # Divide the result by the inverse scale
        t2 = t1 / self.scale
        
        # Add the attention mask to the result
        t3 = t2 + attn_mask
        
        # Apply softmax to the result along the last dimension
        t4 = torch.softmax(t3, dim=-1)
        
        # Perform matrix multiplication between the result and the value tensor
        t5 = t4.matmul(v)  # Shape: (batch_size, num_heads, seq_length, value_dim)

        return t5

# Initializing the model
embed_size = 64  # Size of the embedding
num_heads = 8    # Number of attention heads
model = AttentionModel(embed_size, num_heads)

# Input tensors
batch_size = 2
seq_length = 10
value_dim = 64  # Must match embed_size

query_tensor = torch.randn(batch_size, seq_length, embed_size)
key_tensor = torch.randn(batch_size, seq_length, embed_size)
value_tensor = torch.randn(batch_size, seq_length, value_dim)

# Create an attention mask (for example, a mask of -inf for padding)
attn_mask = torch.zeros(batch_size, seq_length, seq_length) - float('inf')

# Running the model
output = model(query_tensor, key_tensor, value_tensor, attn_mask)
print(output.shape)  # Output shape should be: (batch_size, num_heads, seq_length, value_dim)
