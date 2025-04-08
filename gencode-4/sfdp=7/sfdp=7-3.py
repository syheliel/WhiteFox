import torch
import math

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_size, heads):
        super(AttentionModel, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.values = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.keys = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.queries = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = torch.nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value):
        N = query.shape[0]  # Batch size
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        # Permute the dimensions of the query, key, and value tensors
        q = self.queries(query).permute(0, 2, 1, 3)
        k = self.keys(key).permute(0, 2, 1, 3)
        v = self.values(value).permute(0, 2, 1, 3)

        # Compute the dot product of the query and the transposed key
        div = (q @ k.transpose(-2, -1)) / math.sqrt(self.embed_size)
        div = div.to(torch.float32)  # Convert the result to float32
        
        # Apply softmax to get attention weights
        attn_weight = torch.softmax(div, dim=-1)
        attn_weight = attn_weight.to(torch.float16)  # Convert the result to float16
        
        # Compute the output as the dot product of attention weights and value
        output = attn_weight @ v
        output = output.permute(0, 2, 1, 3)  # Optional: permute back to original dimensions
        output = self.fc_out(output)
        return output

# Initializing the model
embed_size = 64  # Embedding size
heads = 8  # Number of attention heads
model = AttentionModel(embed_size, heads)

# Inputs to the model
query = torch.randn(1, 10, embed_size)  # Batch size 1, sequence length 10
key = torch.randn(1, 10, embed_size)    # Batch size 1, sequence length 10
value = torch.randn(1, 10, embed_size)  # Batch size 1, sequence length 10

# Forward pass
output = model(query, key, value)

print(output.shape)  # Output shape
