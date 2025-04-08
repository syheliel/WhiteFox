import torch
import math

class SelfAttentionModel(torch.nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.values = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.keys = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.queries = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = torch.nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value):
        # Permute the dimensions of the query, key, and value tensors
        q = self.queries(query).permute(0, 2, 1, 3)  # (batch_size, heads, seq_length, embed_size)
        k = self.keys(key).permute(0, 2, 1, 3)      # (batch_size, heads, seq_length, embed_size)
        v = self.values(value).permute(0, 2, 1, 3)  # (batch_size, heads, seq_length, embed_size)

        # Scale the query tensor by the square root of its last dimension size
        q = q / math.sqrt(q.size(-1))  # (batch_size, heads, seq_length, embed_size)
        
        # Perform matrix multiplication between the query tensor and the transposed key tensor
        div = q @ k.transpose(-2, -1)  # (batch_size, heads, seq_length, seq_length)
        
        # Convert the result to float32
        div = div.to(torch.float32)
        
        # Apply softmax to the result along the last dimension
        attn_weight = torch.softmax(div, dim=-1)  # (batch_size, heads, seq_length, seq_length)
        
        # Convert the result to float16
        attn_weight = attn_weight.to(torch.float16)
        
        # Perform matrix multiplication between the attention weights and the value tensor
        out = attn_weight @ v  # (batch_size, heads, seq_length, embed_size)
        
        # Final linear layer
        out = out.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_length, heads, embed_size)
        out = out.view(out.shape[0], out.shape[1], -1)  # (batch_size, seq_length, heads*embed_size)
        return self.fc_out(out)  # (batch_size, seq_length, embed_size)

# Initializing the model
embed_size = 64  # Example embedding size
heads = 4       # Number of attention heads
model = SelfAttentionModel(embed_size, heads)

# Inputs to the model
batch_size = 2
seq_length = 10
input_tensor = torch.randn(batch_size, seq_length, embed_size)  # (batch_size, seq_length, embed_size)

# Forward pass
output = model(input_tensor, input_tensor, input_tensor)

# Display output shape
print(output.shape)  # Should print (batch_size, seq_length, embed_size)
