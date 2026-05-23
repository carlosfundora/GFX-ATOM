import sys

with open('atom/audio/chatterbox/engine.py', 'r') as f:
    content = f.read()

rust_code = """
    @staticmethod
    def _np_rep_penalty(input_ids, scores, penalty):
        if _HAS_RS_CODEC:
            rs_codec.np_rep_penalty(scores, input_ids, penalty)
            return scores
        if input_ids.shape[0] == 1:
            ids = input_ids[0]
            s = scores[0, ids]
            mask = s < 0
            s[mask] *= penalty
            s[~mask] /= penalty
            scores[0, ids] = s
            return scores

        score = np.take_along_axis(scores, input_ids, axis=1)
        mask = score < 0
        score[mask] *= penalty
        score[~mask] /= penalty
        np.put_along_axis(scores, input_ids, score, axis=1)
        return scores
"""

# The original method starts with @staticmethod and def _np_rep_penalty
# Let's do a replace

old_code = """    @staticmethod
    def _np_rep_penalty(input_ids, scores, penalty):
        if input_ids.shape[0] == 1:
            ids = input_ids[0]
            s = scores[0, ids]
            mask = s < 0
            s[mask] *= penalty
            s[~mask] /= penalty
            scores[0, ids] = s
            return scores

        score = np.take_along_axis(scores, input_ids, axis=1)
        mask = score < 0
        score[mask] *= penalty
        score[~mask] /= penalty
        np.put_along_axis(scores, input_ids, score, axis=1)
        return scores"""

if old_code in content:
    content = content.replace(old_code, rust_code.strip('\n'))
    with open('atom/audio/chatterbox/engine.py', 'w') as f:
        f.write(content)
    print("Successfully patched engine.py")
else:
    print("Failed to find _np_rep_penalty in engine.py")
