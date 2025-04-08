import torch

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.query_linear = torch.nn.Linear(embed_size, embed_size)
        self.key_linear = torch.nn.Linear(embed_size, embed_size)
        self.value_linear = torch.nn.Linear(embed_size, embed_size)
        self.num_heads = num_heads
        self.embed_size = embed_size

    def forward(self, query, key, value):
        # Compute the dot product of the query and key tensors
        qk = torch.matmul(query, key.transpose(-2, -1))
        
        # Scale the dot product by the inverse scale
        inv_scale = self.embed_size ** 0.5
        scaled_qk = qk / inv_scale
        
        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)
        
        # Compute the dot product of the softmax output and the value tensor
        output = softmax_qk.matmul(value)
        
        return output

# Initializing the model
embed_size = 64  # Size of the embedding
num_heads = 8    # Number of attention heads
attention_model = AttentionModel(embed_size, num_heads)

# Inputs to the model
batch_size = 2   # Number of samples in a batch
sequence_length = 10  # Length of each input sequence
query = torch.randn(batch_size, sequence_length, embed_size)  # Query tensor
key = torch.randn(batch_size, sequence_length, embed_size)    # Key tensor
value = torch.randn(batch_size, sequence_length, embed_size)  # Value tensor

# Forward pass through the model
output = attention_model(query, key, value)

# Output shape
print("Output shape:", output.shape)
