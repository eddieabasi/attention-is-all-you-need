import torch

from transformer.checkpoint import average_state_dicts, list_checkpoints, load_model, prune_checkpoints, save_checkpoint
from transformer.data import BOS_ID, EOS_ID, collate
from transformer.decode import greedy_decode
from transformer.model import Transformer, TransformerConfig
from transformer.optim import build_optimizer


def tiny(seed=0):
    torch.manual_seed(seed)
    return Transformer(TransformerConfig(vocab_size=20, d_model=16, num_heads=2, d_ff=32, num_encoder_layers=1, num_decoder_layers=1))


def test_greedy_decode_respects_length_cap_and_strips_specials():
    model = tiny()
    batch = collate([([5, 6, 7], [5]), ([8], [8])])
    out = greedy_decode(model, batch.src, max_extra_len=4)
    assert len(out) == 2
    for hyp, src_len in zip(out, (4, 2)):  # src lengths include EOS
        assert len(hyp) <= src_len + 4
        assert BOS_ID not in hyp and EOS_ID not in hyp and 0 not in hyp


def test_greedy_decode_matches_step_by_step_argmax():
    model = tiny().eval()
    batch = collate([([5, 6, 7, 8], [5])])
    (hyp,) = greedy_decode(model, batch.src, max_extra_len=3)
    ys = [BOS_ID]
    for _ in range(len(hyp)):
        logits = model(batch.src, torch.tensor([ys]))[0, -1]
        logits[[0, BOS_ID]] = float("-inf")
        ys.append(int(logits.argmax()))
    assert ys[1:] == hyp


def test_checkpoint_roundtrip_and_averaging(tmp_path):
    models = [tiny(seed) for seed in (1, 2)]
    for step, m in zip((10, 20), models):
        opt, sched = build_optimizer(m, 16, 100)
        save_checkpoint(tmp_path / f"checkpoint_{step}.pt", m, opt, sched, step, {"data": {}}, epoch=0)
    paths = list_checkpoints(tmp_path)
    assert [p.name for p in paths] == ["checkpoint_10.pt", "checkpoint_20.pt"]

    loaded, ckpt = load_model(paths[1])
    assert ckpt["step"] == 20
    for (k, a), b in zip(models[1].state_dict().items(), loaded.state_dict().values()):
        torch.testing.assert_close(a, b)

    avg = average_state_dicts(paths)
    k = "embed.lut.weight"
    torch.testing.assert_close(avg[k], (models[0].state_dict()[k] + models[1].state_dict()[k]) / 2)

    prune_checkpoints(tmp_path, keep=1)
    assert [p.name for p in list_checkpoints(tmp_path)] == ["checkpoint_20.pt"]
