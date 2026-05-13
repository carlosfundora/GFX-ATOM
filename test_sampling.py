import torch
import torch.nn.functional as F

def sample(logits, temperature=1.0, top_k=50):
    logits = logits / temperature
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

# Test on float16 ROCm
logits = torch.randn(1, 8194, dtype=torch.float16, device="cuda:0") * 10
print("Original multinomial:", sample(logits))

def safe_sample(logits, temperature=1.0, top_k=50):
    logits = logits.float() / temperature
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

print("Safe multinomial:", safe_sample(logits))
