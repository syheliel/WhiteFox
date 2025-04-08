import torch

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_size, num_heads, dropout_p):
        super(AttentionModel, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout_p = dropout_p

    def forward(self, query, key, value):
        # Compute attention weights
        attn_weight = torch.bmm(query, key.transpose(1, 2)).softmax(dim=-1)
        # Apply dropout to the attention weights
        attn_weight = torch.nn.functional.dropout(attn_weight, p=self.dropout_p)
        # Compute the output
        output = torch.bmm(attn_weight, value)
        return output

# Create an instance of the model
embed_size = 64   # Embedding size for the attention mechanism
num_heads = 4     # Number of attention heads
dropout_p = 0.1   # Dropout probability
model = AttentionModel(embed_size, num_heads, dropout_p)

# Inputs to the model
batch_size = 2     # Number of samples in a batch
seq_length = 10    # Length of the input sequences
query = torch.randn(batch_size, seq_length, embed_size)  # Query tensor
key = torch.randn(batch_size, seq_length, embed_size)    # Key tensor
value = torch.randn(batch_size, seq_length, embed_size)  # Value tensor

# Forward pass
output = model(query, key, value)
