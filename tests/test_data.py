import random

from transformer.data import BOS_ID, EOS_ID, PAD_ID, TokenBatcher, collate, synthetic_task


def random_examples(n=500, seed=0):
    rng = random.Random(seed)
    return [([5] * rng.randint(1, 40), [6] * rng.randint(1, 40)) for _ in range(n)]


def test_collate_adds_bos_eos_and_pads():
    batch = collate([([7, 8], [9]), ([7], [9, 10, 11])])
    assert batch.src.tolist() == [[7, 8, EOS_ID], [7, EOS_ID, PAD_ID]]
    assert batch.tgt_in.tolist() == [[BOS_ID, 9, PAD_ID, PAD_ID], [BOS_ID, 9, 10, 11]]
    assert batch.tgt_out.tolist() == [[9, EOS_ID, PAD_ID, PAD_ID], [9, 10, 11, EOS_ID]]
    assert batch.num_tgt_tokens == 6


def test_batches_respect_token_budget():
    examples = random_examples()
    for b in TokenBatcher(examples, max_tokens=200).batches():
        assert b.src.numel() <= 200 and b.tgt_in.numel() <= 200


def test_every_example_used_once_per_epoch_and_order_changes():
    examples = random_examples()
    batcher = TokenBatcher(examples, max_tokens=300, seed=1)
    e0, e1 = batcher.batch_indices(0), batcher.batch_indices(1)
    assert sorted(i for b in e0 for i in b) == list(range(len(examples)))
    assert e0 != e1
    assert batcher.batch_indices(0) == e0  # deterministic given seed + epoch


def test_oversized_example_gets_its_own_batch():
    examples = [([5] * 100, [6] * 3), ([5] * 2, [6] * 2)]
    sizes = sorted(len(b) for b in TokenBatcher(examples, max_tokens=50).batch_indices())
    assert sizes == [1, 1]


def test_synthetic_tasks():
    copy = synthetic_task("copy", train_size=200, valid_size=20, test_size=20)
    assert all(s == t for s, t in copy.train)
    rev = synthetic_task("reverse", train_size=200, valid_size=20, test_size=20)
    assert all(s[::-1] == t for s, t in rev.train)
    train_set = {tuple(s) for s, _ in rev.train}
    assert not any(tuple(s) in train_set for s, _ in rev.test)
    assert all(min(s) >= 4 for s, _ in rev.train)  # no special tokens in the data
