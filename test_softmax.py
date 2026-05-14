import torch
import torch.nn.functional as F

logits = torch.tensor([[10.0, -float('Inf'), 5.0]], dtype=torch.float16, device="cuda:0")
probs = F.softmax(logits, dim=-1)
print(probs)

logits = torch.tensor([[100.0, -float('Inf'), 50.0]], dtype=torch.float16, device="cuda:0")
probs = F.softmax(logits, dim=-1)
print(probs)
