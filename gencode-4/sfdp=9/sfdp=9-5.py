import torch
import math

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.values = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.keys = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.queries = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = torch.nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value):
        N, seq_length, _ = query.shape
        
        # Permute the dimensions
        q = self.queries(query).view(N, seq_length, self.heads, self.embed_size // self.heads).permute(0, 2, 1, 3)
        k = self.keys(key).view(N, seq_length, self.heads, self.embed_size // self.heads).permute(0, 2, 1, 3)
        v = self.values(value).view(N, seq_length, self.heads, self.embed_size // self.heads).permute(0, 2, 1, 3)

        # Scale the query tensor
        q = q / math.sqrt(q.size(-1))

        # Matrix multiplication
        div = q @ k.transpose(-2, -1)

        # Convert the result to float32
        div = div.to(torch.float32)

        # Apply softmax
        attn_weight = torch.softmax(div, dim=-1)

        # Convert the result to float16
        attn_weight = attn_weight.to(torch.float16)

        # Matrix multiplication between attention weights and value tensor
        out = attn_weight @ v
        
        # Returning output
        return out.permute(0, 2, 1, 3).contiguous().view(N, seq_length, -1)

# Initializing the model
embed_size = 64  # Embedding size
heads = 8        # Number of attention heads
model = AttentionModel(embed_size, heads)

# Inputs to the model
# Example input tensor shape: (batch_size, sequence_length, embedding_size)
batch_size = 1
sequence_length = 10
x_query = torch.randn(batch_size, sequence_length, embed_size)
x_key = torch.randn(batch_size, sequence_length, embed_size)
x_value = torch.randn(batch_size, sequence_length, embed_size)

# Forward pass
output = model(x_query, x_key, x_value)
