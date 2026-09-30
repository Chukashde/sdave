"""Behaviour-preservation tests: determinism and RNG order on CPU.

These encode the core requirement: the same experiment with the same seed
produces identical cut values. Absolute golden numbers are machine/BLAS
specific, so the portable guard here is run-to-run determinism on one machine
(plus identical weight init for a fixed seed).
"""
import torch

from pigat.config import PigatConfig
from pigat.models import create_model
from pigat.sweeps import run_sweep
from pigat.training import setup_environment


def _cell(tmp_path, model="pigcn", num_heads=2, attn_drop=0.1, epochs=40):
    cfg = PigatConfig()
    cfg.device = "cpu"
    cfg.number_epochs = epochs
    cfg.patience = epochs
    cfg.dropout = 0.2
    cfg.num_heads = num_heads
    cfg.attn_drop = attn_drop
    cfg.output_dir = str(tmp_path)
    df = run_sweep(cfg, "dataset/Gset/G14", models=[model], seeds=[1],
                   problem="maxcut", architectures=[[16]], save_every=1)
    r = df.iloc[0]
    return float(r["cut_value"]), float(r["final_loss"]), float(r["best_loss"])


def test_cutvalue_deterministic_pigcn(tmp_path):
    a = _cell(tmp_path / "a", model="pigcn")
    b = _cell(tmp_path / "b", model="pigcn")
    assert a == b


def test_cutvalue_deterministic_pigatv2(tmp_path):
    a = _cell(tmp_path / "a", model="pigatv2")
    b = _cell(tmp_path / "b", model="pigatv2")
    assert a == b


def test_rng_order_identical_weights():
    """Same seed -> identical model weights and embedding, for both models."""
    cfg = dict(dim_embedding=32, hidden_dim=[16, 16, 16], number_classes=1,
               dropout=0.2, device="cpu", num_heads=2, feat_drop=0.0, attn_drop=0.1)

    def build(model, seed):
        setup_environment(seed)
        net = create_model(model, cfg)
        emb = torch.nn.Embedding(50, cfg["dim_embedding"])
        return net, emb

    for model in ("pigcn", "pigatv2"):
        n1, e1 = build(model, 1)
        n2, e2 = build(model, 1)
        s1, s2 = n1.state_dict(), n2.state_dict()
        assert s1.keys() == s2.keys()
        assert all(torch.equal(s1[k], s2[k]) for k in s1)
        assert torch.equal(e1.weight, e2.weight)
