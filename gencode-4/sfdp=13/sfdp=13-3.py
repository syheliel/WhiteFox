import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(AttentionModel, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.scale = embed_size ** 0.5  # Scale for the attention scores
        self.query_projection = nn.Linear(embed_size, embed_size)
        self.key_projection = nn.Linear(embed_size, embed_size)
        self.value_projection = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, attn_mask):
        # Permute the dimensions of the query, key, and value tensors
        q = self.query_projection(query).permute(0, 2, 1, 3)
        k = self.key_projection(key).permute(0, 2, 1, 3)
        v = self.value_projection(value).permute(0, 2, 1, 3)

        # Perform matrix multiplication between the query tensor and the transposed key tensor
        t1 = torch.matmul(q, k.transpose(-2, -1))

        # Divide the result by the inverse scale
        t2 = t1 / self.scale

        # Add the attention mask to the result
        t3 = t2 + attn_mask

        # Apply softmax to the result along the last dimension
        t4 = t3.softmax(dim=-1)

        # Perform matrix multiplication between the result and the value tensor
        t5 = t4.matmul(v)

        return t5

# Initialize the model
embed_size = 64  # Size of the embedding
num_heads = 8    # Number of attention heads
model = AttentionModel(embed_size, num_heads)

# Inputs to the model
batch_size = 1
seq_length = 10  # Length of the input sequences
num_features = embed_size  # Number of features in the embedding

# Random input tensors for query, key, and value
query = torch.randn(batch_size, seq_length, num_features)
key = torch.randn(batch_size, seq_length, num_features)
value = torch.randn(batch_size, seq_length, num_features)

# Attention mask (for simplicity, using zeros here; in practice, it should be appropriately sized)
attn_mask = torch.zeros(batch_size, seq_length, seq_length)

# Forward pass
output = model(query, key, value, attn_mask)

# Output shape
print(output.shape)  # Should be [1, 10, 64] (batch_size, seq_length, embed_size)
