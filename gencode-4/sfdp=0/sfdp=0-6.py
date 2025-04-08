import torch

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.query_linear = torch.nn.Linear(embed_size, embed_size)
        self.key_linear = torch.nn.Linear(embed_size, embed_size)
        self.value_linear = torch.nn.Linear(embed_size, embed_size)

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

# Initialize the model with specified embedding size and number of heads
embed_size = 64  # Embedding size for queries, keys, and values
num_heads = 8    # Number of attention heads (not used in this simple implementation)
model = AttentionModel(embed_size, num_heads)

# Inputs to the model
batch_size = 1
sequence_length = 10  # Length of the input sequences
query = torch.randn(batch_size, sequence_length, embed_size)  # Query tensor
key = torch.randn(batch_size, sequence_length, embed_size)    # Key tensor
value = torch.randn(batch_size, sequence_length, embed_size)  # Value tensor

# Forward pass through the model
output = model(query, key, value)
