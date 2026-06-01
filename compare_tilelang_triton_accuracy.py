#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from einops import repeat


def unified_reference(A_log, a, dt_bias, q, k, v, b, init_state,
                      state_indices, cu_seqlens, scale, eps=1e-6):
    B_, T_, H_, K_ = k.shape
    HV_ = v.shape[2]
    V_ = v.shape[-1]
    q_ref = repeat(q, "b t h d -> b t (h g) d", g=HV_ // H_).float()
    k_ref = repeat(k, "b t h d -> b t (h g) d", g=HV_ // H_).float()
    q_ref = F.normalize(q_ref, p=2, dim=-1, eps=eps) * scale
    k_ref = F.normalize(k_ref, p=2, dim=-1, eps=eps)

    out = torch.empty((B_, T_, HV_, V_), dtype=torch.float32)
    next_state = init_state.clone().float()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias, beta=1.0, threshold=20.0)
    beta_val = b.float().sigmoid()

    for seq_idx in range(cu_seqlens.numel() - 1):
        bos = int(cu_seqlens[seq_idx])
        eos = int(cu_seqlens[seq_idx + 1])
        si = int(state_indices[seq_idx])
        h = torch.zeros_like(next_state[0]) if si < 0 else next_state[si].clone()
        for tok in range(bos, eos):
            bi, li = tok // T_, tok % T_
            h = h * torch.exp(g.reshape(B_ * T_, HV_)[tok]).view(HV_, 1, 1)
            v_cur = v.reshape(B_ * T_, HV_, -1)[tok].float()
            k_cur = k_ref.reshape(B_ * T_, HV_, K_)[tok]
            q_cur = q_ref.reshape(B_ * T_, HV_, K_)[tok]
            v_cur = v_cur - torch.sum(h * k_cur[:, :, None], dim=1)
            v_cur = v_cur * beta_val.reshape(B_ * T_, HV_)[tok][:, None]
            h = h + k_cur[:, :, None] * v_cur[:, None, :]
            out[bi, li] = torch.sum(h * q_cur[:, :, None], dim=1)
        if si >= 0:
            next_state[si] = h
    return out, next_state


def compare(name, actual, expected,
            mare_thresh=10.0, mere_thresh=2.0, rmse_thresh=2.0,
            small_golden_thresh=1e-3, small_err_thresh=1e-3):
    actual_f = actual.cpu().float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    abs_expected = expected_f.abs()

    rel = diff / (abs_expected + 1e-7)
    mare = rel.max().item()
    mere = rel.mean().item()
    rmse = diff.pow(2).mean().sqrt().item()

    small_mask = (abs_expected < small_golden_thresh) & (diff > small_err_thresh)
    small_val_errors = int(small_mask.sum().item())

    mare_ratio = mare / mare_thresh if mare_thresh > 0 else float("inf")
    mere_ratio = mere / mere_thresh if mere_thresh > 0 else float("inf")
    rmse_ratio = rmse / rmse_thresh if rmse_thresh > 0 else float("inf")

    passed = mare_ratio <= 1.0 and mere_ratio <= 1.0 and rmse_ratio <= 1.0 and small_val_errors == 0
    status = "PASS" if passed else "FAIL"

    print(f"  {name}: [{status}]")
    print(f"    MARE           = {mare:.6e}  (ratio={mare_ratio:.4f}, thresh={mare_thresh})")
    print(f"    MERE           = {mere:.6e}  (ratio={mere_ratio:.4f}, thresh={mere_thresh})")
    print(f"    RMSE           = {rmse:.6e}  (ratio={rmse_ratio:.4f}, thresh={rmse_thresh})")
    print(f"    SmallVal errors= {small_val_errors}  (|golden|<{small_golden_thresh}, |diff|>{small_err_thresh})")

    flat_idx = int(diff.flatten().argmax())
    idx = tuple(int(i) for i in torch.unravel_index(torch.tensor(flat_idx), diff.shape))
    print(f"    MaxDiff at     = {idx}")
    print(f"    actual         = {actual_f.flatten()[flat_idx].item():.8e}")
    print(f"    expected       = {expected_f.flatten()[flat_idx].item():.8e}")
    print(f"    abs_diff       = {diff.flatten()[flat_idx].item():.8e}")


def main():
    torch.manual_seed(42)
    B, T, H, HV, K, V = 8, 1, 16, 32, 128, 128
    dtype = torch.bfloat16
    device = "npu"
    scale = K ** -0.5

    q = torch.randn(B, T, H, K, dtype=dtype)
    k = torch.randn(B, T, H, K, dtype=dtype)
    v = torch.randn(B, T, HV, V, dtype=dtype)
    a = torch.randn(B, T, HV, dtype=dtype)
    b = torch.randn(B, T, HV, dtype=dtype)
    A_log = torch.randn(HV, dtype=torch.float32)
    dt_bias = torch.randn(HV, dtype=torch.float32)
    initial_state = torch.randn(B, HV, K, V, dtype=torch.float32)
    state_indices = torch.arange(B, dtype=torch.int32)
    cu_seqlens = torch.arange(B + 1, dtype=torch.int32) * T

    print("=== Computing unified float32 reference ===")
    ref_out, ref_state = unified_reference(
        A_log, a, dt_bias, q, k, v, b, initial_state,
        state_indices, cu_seqlens, scale,
    )
    # ref_out_bf16init, ref_state_bf16init = unified_reference(
    #     A_log, a, dt_bias, q, k, v, b, initial_state.to(dtype),
    #     state_indices, cu_seqlens, scale,
    # )

    print("=== Running Triton kernel ===")
    from third_party.torch_npu_ops.triton_npu.triton_src.test_fused_sigmoid_gating_delta_rule_update import (
        fused_sigmoid_gating_delta_rule_update,
    )
    state_tri = initial_state.clone().to(device)
    tri_out = fused_sigmoid_gating_delta_rule_update(
        A_log.to(device), a.to(device), dt_bias.to(device),
        q.to(device), k.to(device), v.to(device), b.to(device),
        state_tri, state_indices.to(device), cu_seqlens.to(device),
        scale, True,
    )

    print("=== Running TileLang kernel ===")
    import tilelang
    from xllm.compiler.tilelang.targets.ascend.kernels.fused_sigmoid_gating_delta_rule import (
        fused_sigmoid_gating_delta_rule_kernel_jit,
        _auto_block_v,
        SOFTPLUS_THRESHOLD,
    )
    tilelang.disable_cache()

    total_tokens = B * T
    padding = 64
    total_tokens_padded = total_tokens + padding

    q_tl = torch.zeros(total_tokens_padded, H, K, dtype=dtype, device=device)
    k_tl = torch.zeros(total_tokens_padded, H, K, dtype=dtype, device=device)
    v_tl = torch.zeros(total_tokens_padded, HV, V, dtype=dtype, device=device)
    a_tl = torch.zeros(total_tokens_padded, HV, dtype=dtype, device=device)
    b_tl = torch.zeros(total_tokens_padded, HV, dtype=dtype, device=device)
    q_tl[:total_tokens] = q.reshape(total_tokens, H, K).to(device)
    k_tl[:total_tokens] = k.reshape(total_tokens, H, K).to(device)
    v_tl[:total_tokens] = v.reshape(total_tokens, HV, V).to(device)
    a_tl[:total_tokens] = a.reshape(total_tokens, HV).to(device)
    b_tl[:total_tokens] = b.reshape(total_tokens, HV).to(device)
    # init_state_tl = initial_state.to(device).to(dtype)
    init_state_tl = initial_state.to(device)

    ker = fused_sigmoid_gating_delta_rule_kernel_jit(
        nk=H, nv=HV, dk=K, dv=V, block_v=_auto_block_v(V),
        max_num_seqs=B, use_qk_l2norm=1, softplus_beta=1.0,
        dtype="bf16", accum_dtype="float",
    )
    # print(ker.get_kernel_source())
    tl_out, tl_state = ker(
        A_log.to(device), a_tl, dt_bias.to(device),
        q_tl, k_tl, v_tl, b_tl, init_state_tl,
        state_indices.to(device), cu_seqlens.to(device),
        1.0, scale, 1, SOFTPLUS_THRESHOLD,
    )
    print(tl_state[2, 6, :, 64:128])
    t = tl_state[2, 6, :, 64:128]
    all_zero_rows = (t.abs().sum(dim=1) == 0).nonzero()
    print(f"first all-zero row: {all_zero_rows[0].item() if len(all_zero_rows) else 'none'}")

    print("\n" + "=" * 60)
    print("Triton vs Reference")
    print("=" * 60)
    compare("output", tri_out, ref_out)
    compare("state", state_tri, ref_state)

    print("\n" + "=" * 60)
    print("TileLang vs Reference (bf16 init_state)")
    print("=" * 60)
    tl_out_reshaped = tl_out[:total_tokens].reshape(B, T, HV, V)
    compare("output", tl_out_reshaped, ref_out)
    compare("state", tl_state[:B].float(), ref_state)

    print("\n" + "=" * 60)
    print("TileLang vs Triton")
    print("=" * 60)
    compare("output", tl_out_reshaped, tri_out.cpu())
    compare("state", tl_state[:B].float(), state_tri.cpu().float())


if __name__ == "__main__":
    main()
