import sys
import numpy as np
from unittest.mock import MagicMock

sys.modules['onnxruntime'] = MagicMock()

# Instead of importing the whole engine, let's just test our new preallocation logic in isolation
def _np_rep_penalty(input_ids, scores, penalty):
    # filter out indices that are greater than vocab size for our mock
    vocab_size = scores.shape[1]
    valid_ids = input_ids.copy()
    valid_ids[valid_ids >= vocab_size] = 0
    score = np.take_along_axis(scores, valid_ids, axis=1)
    score = np.where(score < 0, score * penalty, score / penalty)
    out = scores.copy()
    np.put_along_axis(out, valid_ids, score, axis=1)
    return out

def rep_penalty_fn(ids, scores):
    return _np_rep_penalty(ids, scores, 1.2)

batch_size = 1
max_tokens = 5
inputs_embeds = np.random.randn(1, 3, 64)
seq_len = inputs_embeds.shape[1]

generate_tokens = np.zeros((batch_size, 1 + max_tokens), dtype=np.int64)
generate_tokens[0, 0] = 100 # START_SPEECH_TOKEN
gen_idx = 1

attention_mask = np.ones((batch_size, seq_len + max_tokens), dtype=np.int64)

for i in range(max_tokens):
    if i == 0:
        cur_embeds = inputs_embeds
    else:
        cur_embeds = np.random.randn(1, 1, 64) # mocked embed_single_token

    current_seq_len = seq_len + i
    cur_attention_mask = attention_mask[:, :current_seq_len]

    # Mock LLM run
    logits = np.random.randn(1, 1, 200) # mock logits
    logits = logits[:, -1, :]

    cur_gen_tokens = generate_tokens[:, :gen_idx]
    logits = rep_penalty_fn(cur_gen_tokens, logits)

    next_token = np.argmax(logits, axis=-1, keepdims=True).astype(np.int64)
    generate_tokens[:, gen_idx] = next_token[:, 0]
    gen_idx += 1

    if (next_token.flatten() == 200).all(): # STOP_SPEECH_TOKEN
        break

# Strip start/stop
tokens = generate_tokens[:, 1:gen_idx]
if tokens[0, -1] == 200:
    tokens = tokens[:, :-1]

print("Success! Tokens shape:", tokens.shape)
