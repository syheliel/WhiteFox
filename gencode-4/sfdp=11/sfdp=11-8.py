import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, dropout_p):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout_p = dropout_p
        self.d_model = d_model

    def forward(self, query, key, value, inv_scale_factor):
        q = query.permute(0, 2, 1, 3)  # Permute the dimensions of the query tensor
        k = key.permute(0, 2, 1, 3)      # Permute the dimensions of the key tensor
        v = value.permute(0, 2, 1, 3)    # Permute the dimensions of the value tensor

        t1 = torch.matmul(q, k.transpose(-2, -1))  # Matrix multiplication
        t2 = t1.div(inv_scale_factor)                # Divide by inverse scale factor
        t3 = t2.softmax(dim=-1)                      # Apply softmax
        t4 = F.dropout(t3, p=self.dropout_p)        # Apply dropout

        output = t4.matmul(v)                        # Final matrix multiplication
        return output

# Initializing the model
d_model = 64  # Model dimension
dropout_p = 0.1  # Dropout probability
inv_scale_factor = d_model ** 0.5  # Inverse scale factor

attention_model = ScaledDotProductAttention(d_model, dropout_p)

# Inputs to the model (batch_size, seq_len, num_heads, d_model)
batch_size = 1
seq_len = 10
num_heads = 4

# Generating input tensors
query_tensor = torch.randn(batch_size, seq_len, num_heads, d_model)
key_tensor = torch.randn(batch_size, seq_len, num_heads, d_model)
value_tensor = torch.randn(batch_size, seq_len, num_heads, d_model)

# Getting the output from the model
output = attention_model(query_tensor, key_tensor, value_tensor, inv_scale_factor)

# Display output
print(output.shape)  # Should match (batch_size, seq_len, num_heads, d_model)
