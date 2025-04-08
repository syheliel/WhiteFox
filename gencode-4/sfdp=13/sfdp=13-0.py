import torch
import torch.nn as nn

class SelfAttentionModel(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(SelfAttentionModel, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.scale = embed_size ** 0.5

    def forward(self, query, key, value, attn_mask):
        # Permute the dimensions of the query, key, and value tensors
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)
        
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

# Initializing the model
embed_size = 64  # Example embedding size
num_heads = 8    # Example number of heads
model = SelfAttentionModel(embed_size, num_heads)

# Inputs to the model
batch_size = 1
sequence_length = 10
num_values = 12  # Number of value features

# Creating example input tensors with appropriate dimensions
query = torch.randn(batch_size, sequence_length, num_values, embed_size)
key = torch.randn(batch_size, sequence_length, num_values, embed_size)
value = torch.randn(batch_size, sequence_length, num_values, embed_size)
attn_mask = torch.zeros(batch_size, sequence_length, sequence_length)  # Attention mask

# Forward pass through the model
output = model(query, key, value, attn_mask)

# Output shape
print(output.shape)  # Should be (batch_size, sequence_length, num_values, embed_size)
