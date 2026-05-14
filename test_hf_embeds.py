import torch
from transformers import AutoModelForCausalLM, AutoConfig

backbone_dir = "/home/local/Projects/models/huggingface/models--vladislavbro--llama_backbone_0.5/snapshots/a6c48da4d993a2058b95a8c3e2178da29f603f3e"
model = AutoModelForCausalLM.from_pretrained(backbone_dir, torch_dtype=torch.float16, device_map="cuda:0")

# dummy inputs_embeds
inputs_embeds = torch.randn(1, 10, model.config.hidden_size, dtype=torch.float16, device="cuda:0")

out1 = model(inputs_embeds=inputs_embeds, use_cache=True)
pkv = out1.past_key_values

next_embed = torch.randn(1, 1, model.config.hidden_size, dtype=torch.float16, device="cuda:0")
out2 = model(inputs_embeds=next_embed, past_key_values=pkv, use_cache=True)

print("First pass logits shape:", out1.logits.shape)
print("Second pass logits shape:", out2.logits.shape)
print("Second pass logits:", out2.logits[0, 0, :5])

# Now compare with explicit attention mask and position ids
attention_mask = torch.ones(1, 10, dtype=torch.long, device="cuda:0")
out1_explicit = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=True)
pkv_exp = out1_explicit.past_key_values

attention_mask_next = torch.ones(1, 11, dtype=torch.long, device="cuda:0")
position_ids_next = torch.tensor([[10]], dtype=torch.long, device="cuda:0")
out2_explicit = model(
    inputs_embeds=next_embed, 
    past_key_values=pkv_exp, 
    attention_mask=attention_mask_next, 
    position_ids=position_ids_next, 
    use_cache=True
)
print("Explicit mask second pass logits:", out2_explicit.logits[0, 0, :5])
print("Difference:", (out2.logits - out2_explicit.logits).abs().max())
