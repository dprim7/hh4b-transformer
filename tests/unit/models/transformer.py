import math
import numpy as np
import torch
import pytest

from hh4b_transformer.models.transformer import HH4bTransformer


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    torch.manual_seed(0)
    np.random.seed(0)


def _make_dummy(batch_size: int = 2, num_tokens: int = 7, feature_dim: int = 8):
    x = torch.randn(batch_size, num_tokens, feature_dim, dtype=torch.float32)
    # first k tokens valid, rest padded
    mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool)
    for b in range(batch_size):
        valid = 1 + (b % (num_tokens - 1))  # at least 1 valid
        mask[b, :valid] = True
    return x, mask


def test_multiclass_shapes() -> None:
    B, N, F = 3, 5, 8
    x, mask = _make_dummy(B, N, F)

    C = 11
    m_mc = HH4bTransformer(
        feature_dim=F, d_model=32, n_heads=4, n_layers=2, dropout=0.0, num_classes=C
    ).eval()
    y_mc = m_mc(x, mask)
    assert y_mc.shape == (B, C)
    assert torch.isfinite(y_mc).all()


def test_mask_excludes_padding_tokens() -> None:
    B, N, F = 2, 6, 8
    x, mask = _make_dummy(B, N, F)

    model = HH4bTransformer(
        feature_dim=F, d_model=32, n_heads=4, n_layers=1, dropout=0.0, num_classes=5, pooling="mean"
    ).eval()

    y1 = model(x.clone(), mask)

    # Change padded positions drastically; output should be (nearly) invariant
    x2 = x.clone()
    x2[~mask] = 1000.0
    y2 = model(x2, mask)

    assert torch.allclose(y1, y2, atol=1e-5)


def test_pairwise_bias_path_changes_output() -> None:
    B, N, F, K = 2, 5, 8, 4
    x, mask = _make_dummy(B, N, F)

    model = HH4bTransformer(
        feature_dim=F,
        d_model=32,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
        num_classes=7,
        use_pair_bias=True,
        pair_bias_dim=K,
    ).eval()

    bias_zeros = torch.zeros(B, N, N, K)
    bias_rand = torch.randn(B, N, N, K) * 0.01

    y0 = model(x, mask, attention_bias=bias_zeros)
    y1 = model(x, mask, attention_bias=bias_rand)
    # With non-zero biases, outputs should differ
    assert not torch.allclose(y0, y1)


def test_dropout_train_vs_eval_behavior() -> None:
    B, N, F = 2, 6, 8
    x, mask = _make_dummy(B, N, F)

    model = HH4bTransformer(
        feature_dim=F, d_model=32, n_heads=4, n_layers=2, dropout=0.5, num_classes=3
    )

    model.train()
    y_train_1 = model(x, mask)
    y_train_2 = model(x, mask)
    assert not torch.allclose(y_train_1, y_train_2)

    model.eval()
    y_eval_1 = model(x, mask)
    y_eval_2 = model(x, mask)
    assert torch.allclose(y_eval_1, y_eval_2)


def test_backward_step_runs() -> None:
    B, N, F, C = 2, 4, 8, 9
    x, mask = _make_dummy(B, N, F)

    model = HH4bTransformer(
        feature_dim=F, d_model=32, n_heads=4, n_layers=1, dropout=0.1, num_classes=C
    ).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    targets = torch.randint(0, C, (B,))

    logits = model(x, mask)
    loss = torch.nn.CrossEntropyLoss()(logits, targets)
    assert math.isfinite(loss.item())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    # ensure some gradient flowed
    any_grad = False
    for p in model.parameters():
        if p.grad is not None and torch.isfinite(p.grad).all():
            any_grad = True
            break
    assert any_grad
