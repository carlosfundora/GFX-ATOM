import torch
import triton
import triton.language as tl


@triton.jit
def _pack_q_and_weights_kernel(
    q_ptr,
    weights_ptr,
    decode_lens_ptr,
    cu_seq_lens_ptr,
    out_q_ptr,
    out_weights_ptr,
    stride_q_tok,
    stride_q_head,
    stride_q_dim,
    stride_qout_bs,
    stride_qout_seq,
    stride_qout_head,
    stride_qout_dim,
    max_seq_len,
    head_dim: tl.constexpr,
    n_heads: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    pid_bs = tl.program_id(0)
    pid_seq = tl.program_id(1)

    seq_len = tl.load(decode_lens_ptr + pid_bs)
    if pid_seq < seq_len:
        cu_seq_len = tl.load(cu_seq_lens_ptr + pid_bs)
        in_idx = cu_seq_len + pid_seq

        # Pack Q
        for head_idx in range(n_heads):
            dim_offsets = tl.arange(0, BLOCK_DIM)
            mask = dim_offsets < head_dim

            offset_in = (
                in_idx * stride_q_tok
                + head_idx * stride_q_head
                + dim_offsets * stride_q_dim
            )
            offset_out = (
                pid_bs * stride_qout_bs
                + pid_seq * stride_qout_seq
                + head_idx * stride_qout_head
                + dim_offsets * stride_qout_dim
            )

            val = tl.load(q_ptr + offset_in, mask=mask, other=0.0)
            tl.store(out_q_ptr + offset_out, val, mask=mask)

        # Pack Weights
        if out_weights_ptr is not None:
            w = tl.load(weights_ptr + in_idx)
            tl.store(out_weights_ptr + pid_bs * max_seq_len + pid_seq, w)
    else:
        # Pad Q
        for head_idx in range(n_heads):
            dim_offsets = tl.arange(0, BLOCK_DIM)
            mask = dim_offsets < head_dim

            offset_out = (
                pid_bs * stride_qout_bs
                + pid_seq * stride_qout_seq
                + head_idx * stride_qout_head
                + dim_offsets * stride_qout_dim
            )
            tl.store(out_q_ptr + offset_out, 0.0, mask=mask)

        # Pad Weights
        if out_weights_ptr is not None:
            tl.store(out_weights_ptr + pid_bs * max_seq_len + pid_seq, 0.0)


def triton_pack_q_and_weights(q_fp8, weights, decode_lens):
    batch_size = decode_lens.shape[0]
    max_seq_len = torch.max(decode_lens).item()

    cu_seq_lens = torch.zeros(
        batch_size + 1, dtype=torch.int32, device=decode_lens.device
    )
    cu_seq_lens[1:] = torch.cumsum(decode_lens, dim=0)

    padded_q = torch.empty(
        (batch_size, max_seq_len, q_fp8.shape[1], q_fp8.shape[2]),
        dtype=q_fp8.dtype,
        device=q_fp8.device,
    )

    padded_weights = (
        torch.empty(
            (batch_size * max_seq_len,), dtype=weights.dtype, device=weights.device
        )
        if weights is not None
        else None
    )

    n_heads = q_fp8.shape[1]
    head_dim = q_fp8.shape[2]
    BLOCK_DIM = triton.next_power_of_2(head_dim)

    grid = (batch_size, max_seq_len)
    _pack_q_and_weights_kernel[grid](
        q_fp8,
        weights,
        decode_lens,
        cu_seq_lens,
        padded_q,
        padded_weights,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        padded_q.stride(0),
        padded_q.stride(1),
        padded_q.stride(2),
        padded_q.stride(3),
        max_seq_len,
        head_dim,
        n_heads,
        BLOCK_DIM=BLOCK_DIM,
    )

    return padded_q, padded_weights


@triton.jit
def _unpack_topk_kernel(
    input_ptr,
    decode_lens_ptr,
    cu_seq_lens_ptr,
    output_ptr,
    stride_in_bs,
    stride_in_seq,
    stride_in_dim,
    stride_out_tok,
    stride_out_dim,
    dim_size: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    pid_bs = tl.program_id(0)
    pid_seq = tl.program_id(1)

    seq_len = tl.load(decode_lens_ptr + pid_bs)
    if pid_seq < seq_len:
        cu_seq_len = tl.load(cu_seq_lens_ptr + pid_bs)
        out_idx = cu_seq_len + pid_seq

        dim_offsets = tl.arange(0, BLOCK_DIM)
        mask = dim_offsets < dim_size

        offset_in = (
            pid_bs * stride_in_bs
            + pid_seq * stride_in_seq
            + dim_offsets * stride_in_dim
        )
        offset_out = out_idx * stride_out_tok + dim_offsets * stride_out_dim

        val = tl.load(input_ptr + offset_in, mask=mask)
        tl.store(output_ptr + offset_out, val, mask=mask)


def triton_unpack_topk(padded_tensor, decode_lens):
    batch_size, max_seq_len, dim_size = padded_tensor.shape
    num_tokens = torch.sum(decode_lens).item()

    cu_seq_lens = torch.zeros(
        batch_size + 1, dtype=torch.int32, device=decode_lens.device
    )
    cu_seq_lens[1:] = torch.cumsum(decode_lens, dim=0)

    unpacked_tensor = torch.empty(
        (num_tokens, dim_size), dtype=padded_tensor.dtype, device=padded_tensor.device
    )

    BLOCK_DIM = triton.next_power_of_2(dim_size)

    grid = (batch_size, max_seq_len)
    _unpack_topk_kernel[grid](
        padded_tensor,
        decode_lens,
        cu_seq_lens,
        unpacked_tensor,
        padded_tensor.stride(0),
        padded_tensor.stride(1),
        padded_tensor.stride(2),
        unpacked_tensor.stride(0),
        unpacked_tensor.stride(1),
        dim_size,
        BLOCK_DIM=BLOCK_DIM,
    )
    return unpacked_tensor
