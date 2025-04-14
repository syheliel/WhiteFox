import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, attn_mask):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        bs = q.size(0)
        k_len = k.size(-2)
        scores = q @ k.transpose(-2, -1)
        scores = scores.div(math.sqrt(q.size(-1)))
        attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)
        return torch.softmax(scores.masked_fill(attn_mask, -float('inf')), dim=-1) @ v
```

```yaml
- nn.Linear
- nn.MultiheadAttention
- nn.functional.scaled_dot_product_attention
- nn.functional.softmax
- nn.functional.dropout\