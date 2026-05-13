import torch
import torch.nn.functional as F

def sample(logits, temperature=1.0, top_k=50):
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    
    logits = logits / temperature
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

logits = torch.randn(1, 8194)
print(sample(logits, temperature=1.0, top_k=50))
