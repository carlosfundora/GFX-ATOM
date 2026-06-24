import re

with open('atom/audio/chatterbox/engine.py', 'r') as f:
    content = f.read()

# We need to optimize _generate_onnx_cpu by avoiding np.concatenate on attention_mask every step
# Let's see how attention_mask is constructed.
# Originally:
#             if i == 0:
#                 cur_embeds = inputs_embeds
#                 seq_len = cur_embeds.shape[1]
#                 attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
#             else:
#                 cur_embeds = self.service.embed_single_token(
#                     next_token,
#                     exaggeration=exaggeration,
#                 )
#                 attention_mask = np.concatenate(
#                     [attention_mask, np.ones((batch_size, 1), dtype=np.int64)], axis=1
#                 )
#
# A better way is to pre-allocate it if we know the size, or keep a single array of ones and slice it.
# Actually, since it's just ones, we can just do:
# attention_mask = np.ones((batch_size, seq_len + i), dtype=np.int64)

patch = """<<<<<<< SEARCH
        for i in range(max_tokens):
            if i == 0:
                cur_embeds = inputs_embeds
                seq_len = cur_embeds.shape[1]
                attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
            else:
                cur_embeds = self.service.embed_single_token(
                    next_token,
                    exaggeration=exaggeration,
                )
                attention_mask = np.concatenate(
                    [attention_mask, np.ones((batch_size, 1), dtype=np.int64)], axis=1
                )
=======
        seq_len = inputs_embeds.shape[1]
        for i in range(max_tokens):
            if i == 0:
                cur_embeds = inputs_embeds
            else:
                cur_embeds = self.service.embed_single_token(
                    next_token,
                    exaggeration=exaggeration,
                )

            # Avoid np.concatenate overhead
            attention_mask = np.ones((batch_size, seq_len + i), dtype=np.int64)
>>>>>>> REPLACE"""

with open('patch_onnx.patch', 'w') as f:
    f.write(patch)
