import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModel(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(AttentionModel, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size must be divisible by num_heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x, attn_mask):
        N, seq_length, _ = x.shape
        
        # Linear projections
        values = self.values(x)  # (N, seq_length, embed_size)
        keys = self.keys(x)      # (N, seq_length, embed_size)
        queries = self.queries(x) # (N, seq_length, embed_size)

        # Split into heads
        values = values.view(N, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        keys = keys.view(N, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        queries = queries.view(N, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute the dot product and scale
        qk = (queries @ keys.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (N, num_heads, seq_length, seq_length)
        qk += attn_mask  # Adding attention mask

        attn_weights = F.softmax(qk, dim=-1)  # Apply softmax
        output = attn_weights @ values  # Compute output as weighted sum
        output = output.permute(0, 2, 1, 3).contiguous().view(N, seq_length, self.embed_size)  # Reshape back

        return self.fc_out(output)

# Example usage
embed_size = 16  # For example
num_heads = 4
seq_length = 10  # Length of the input sequence
batch_size = 2   # Number of sequences in a batch

# Create the model
model = AttentionModel(embed_size, num_heads)

# Example input tensor: (batch_size, seq_length, embed_size)
x = torch.randn(batch_size, seq_length, embed_size)

# Example attention mask: (batch_size, 1, seq_length, seq_length)
attn_mask = torch.zeros(batch_size, 1, seq_length, seq_length)  # No masking for this example

# Get the output from the model
output = model(x, attn_mask)
