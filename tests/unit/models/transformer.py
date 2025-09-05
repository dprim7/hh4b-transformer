import torch
import pytest

from hh4b_transformer.models.transformer import HH4bTransformer, TransformerBlock


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    torch.manual_seed(42)


def test_output_shapes() -> None:
    """Test multiclass output shapes are correct."""
    B, N, F = 2, 6, 8
    x = torch.randn(B, N, F)
    mask = torch.ones(B, N, dtype=torch.bool)  # all valid

    # Multiclass
    C = 138
    model_mc = HH4bTransformer(
        feature_dim=F, d_model=32, n_heads=4, n_layers=1, num_classes=C, dropout=0.0
    ).eval()
    y_mc = model_mc(x, mask)
    assert y_mc.shape == (B, C), f"Expected (B, C), got {y_mc.shape}"
    assert torch.isfinite(y_mc).all()


def test_mask_invariance() -> None:
    """Padding tokens should not affect output."""
    B, N, F = 2, 4, 8
    x = torch.randn(B, N, F)
    mask = torch.tensor([[True, True, False, False], [True, False, False, False]], dtype=torch.bool)

    model = HH4bTransformer(
        feature_dim=F, d_model=32, n_heads=4, n_layers=1, pooling="mean", dropout=0.0
    ).eval()

    y1 = model(x, mask)

    # Corrupt padding positions
    x_corrupt = x.clone()
    x_corrupt[~mask] = 999.0
    y2 = model(x_corrupt, mask)

    assert torch.allclose(y1, y2, atol=1e-5), "Padding corruption changed output"


def test_pooling_methods_differ() -> None:
    """CLS and mean pooling should give different results."""
    B, N, F = 2, 5, 8
    x = torch.randn(B, N, F)
    mask = torch.ones(B, N, dtype=torch.bool)

    # Create models with identical weights but different pooling
    model_cls = HH4bTransformer(
        feature_dim=F, d_model=32, n_heads=4, pooling="cls", dropout=0.0
    ).eval()
    model_mean = HH4bTransformer(
        feature_dim=F, d_model=32, n_heads=4, pooling="mean", dropout=0.0
    ).eval()

    # Copy weights to ensure only pooling differs
    model_mean.load_state_dict(model_cls.state_dict())

    y_cls = model_cls(x, mask)
    y_mean = model_mean(x, mask)

    assert not torch.allclose(
        y_cls, y_mean, atol=1e-3
    ), "CLS and mean pooling gave identical results"


def test_mask_edge_cases() -> None:
    """Test edge cases: all valid, single valid token."""
    B, N, F = 2, 4, 8
    x = torch.randn(B, N, F)

    model = HH4bTransformer(
        feature_dim=F, d_model=32, n_heads=4, pooling="mean", dropout=0.0
    ).eval()

    # All tokens valid
    mask_all = torch.ones(B, N, dtype=torch.bool)
    y_all = model(x, mask_all)
    assert torch.isfinite(y_all).all()

    # Single token valid per batch
    mask_one = torch.zeros(B, N, dtype=torch.bool)
    mask_one[:, 0] = True
    y_one = model(x, mask_one)
    assert torch.isfinite(y_one).all()

    # Results should differ
    assert not torch.allclose(y_all, y_one)


def test_pairwise_bias_affects_attention() -> None:
    """Test that pairwise bias works at the TransformerBlock level."""

    B, L, D, H, K = 2, 5, 32, 4, 4
    x = torch.randn(B, L, D)

    # Test TransformerBlock directly (where we know it works)
    block = TransformerBlock(D, H, dropout=0.0, use_pair_bias=True, pair_bias_dim=K).eval()

    # Initialize projection with known weights
    with torch.no_grad():
        block.pair_projection[0].weight.fill_(1.0)
        block.pair_projection[0].bias.fill_(0.0)

    # Test zero bias equals no bias
    bias_zero = torch.zeros(B, L, L, K)
    out_no_bias = block(x, None, None)
    out_zero_bias = block(x, None, bias_zero)
    assert torch.allclose(out_no_bias, out_zero_bias, atol=1e-6), "Zero bias should equal no bias"

    # Non-uniform bias should change attention output
    bias_nonuniform = torch.randn(B, L, L, K) * 2.0
    out_biased = block(x, None, bias_nonuniform)

    diff = (out_zero_bias - out_biased).abs().max().item()
    assert not torch.allclose(
        out_zero_bias, out_biased, atol=1e-3
    ), f"Pairwise bias had no effect at block level. Diff: {diff:.6f}"


def test_dropout_determinism() -> None:
    """Dropout should be stochastic in train, deterministic in eval."""
    B, N, F = 2, 4, 8
    x = torch.randn(B, N, F)
    mask = torch.ones(B, N, dtype=torch.bool)

    model = HH4bTransformer(feature_dim=F, d_model=32, n_heads=4, dropout=0.3)

    # Train mode: stochastic
    model.train()
    y1 = model(x, mask)
    y2 = model(x, mask)
    assert not torch.allclose(y1, y2), "Dropout should be stochastic in train mode"

    # Eval mode: deterministic
    model.eval()
    y3 = model(x, mask)
    y4 = model(x, mask)
    assert torch.allclose(y3, y4), "Dropout should be deterministic in eval mode"


def test_crossentropy_loss_compatibility() -> None:
    """Model outputs should work with CrossEntropyLoss."""
    B, N, F, C = 3, 5, 8, 7
    x = torch.randn(B, N, F)
    mask = torch.ones(B, N, dtype=torch.bool)
    targets = torch.randint(0, C, (B,))

    model = HH4bTransformer(feature_dim=F, d_model=32, n_heads=4, num_classes=C, dropout=0.0)
    logits = model(x, mask)

    # Should not raise and should be finite
    loss = torch.nn.CrossEntropyLoss()(logits, targets)
    assert torch.isfinite(loss), "CrossEntropyLoss produced non-finite value"

    # Test backward pass
    loss.backward()
    # Check that some parameters got gradients
    has_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert has_grad, "No finite gradients found after backward pass"


def test_parameter_count_reasonable() -> None:
    """Sanity check that parameter count is reasonable."""
    model = HH4bTransformer(feature_dim=16, d_model=64, n_heads=8, n_layers=2, num_classes=138)
    param_count = sum(p.numel() for p in model.parameters())

    # Should be > 1K but < 50M for this config
    assert 1000 < param_count < 50_000_000, f"Parameter count {param_count} seems unreasonable"
