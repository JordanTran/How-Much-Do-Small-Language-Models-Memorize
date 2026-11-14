import json

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from model.gpt import GPT, PretrainedGPT

TABLE_1 = [
    (1, 32, 8.04e4),
    (1, 64, 1.85e5),
    (1, 128, 4.69e5),
    (1, 256, 1.33e6),
    (2, 32, 9.31e4),
    (2, 64, 2.35e5),
    (2, 128, 6.67e5),
    (2, 256, 2.12e6),
    (4, 32, 1.18e5),
    (4, 64, 3.35e5),
    (4, 128, 1.06e6),
    (4, 256, 3.70e6),
    (8, 32, 1.69e5),
    (8, 64, 5.35e5),
    (8, 128, 1.86e6),
    (8, 256, 6.86e6),
]


@pytest.mark.parametrize("n_layer, d_model, paper", TABLE_1)
def test_params_match_table_1(n_layer, d_model, paper):
    """Synthetic-data models match the parameter counts in the paper's Table 1."""
    n = GPT(n_layer, d_model, vocab_size=2048).n_params()
    assert f"{n:.2e}" == f"{paper:.2e}"


@pytest.mark.parametrize(
    "d_model, paper", [(32, 1_711_968), (64, 3_620_544), (128, 8_027_520), (256, 19_200_768)]
)
def test_params_match_figure_9_exactly(d_model, paper):
    """Text models (8 layers, GPT-2 vocab) match the exact counts in the paper's Figure 9."""
    assert GPT(8, d_model, vocab_size=50257).n_params() == paper


@pytest.mark.parametrize(
    "seq_len, paper", [(4, 6.59e5), (8, 6.60e5), (16, 6.61e5), (32, 6.63e5), (256, 6.92e5)]
)
def test_params_match_table_3(seq_len, paper):
    """Position embeddings have exactly seq_len rows, matching Table 3 (2 layers, d=128)."""
    n = GPT(2, 128, vocab_size=2048, seq_len=seq_len).n_params()
    assert f"{n:.2e}" == f"{paper:.2e}"


@pytest.mark.parametrize(
    "vocab_size, paper", [(128, 4.21e5), (512, 4.71e5), (1024, 5.36e5), (4096, 9.29e5)]
)
def test_params_match_table_4(vocab_size, paper):
    """The embedding has exactly vocab_size rows, matching Table 4 (2 layers, d=128)."""
    n = GPT(2, 128, vocab_size=vocab_size).n_params()
    assert f"{n:.2e}" == f"{paper:.2e}"


def test_embeddings_tied():
    """The output layer shares its weights with the input embedding."""
    model = GPT(2, 64, vocab_size=2048)
    assert model.gpt2.lm_head.weight is model.gpt2.transformer.wte.weight


def test_head_count():
    """Each attention head has dimension 32."""
    heads = [GPT(1, d, vocab_size=2048).gpt2.config.n_head for d in (32, 64, 128, 256)]
    assert heads == [1, 2, 4, 8]


def test_logits_cover_every_token():
    """The model outputs a prediction for all seq_len tokens, including the first."""
    x = torch.randint(0, 2048, (3, 64))
    assert GPT(2, 64, vocab_size=2048)(x).shape == (3, 64, 2048)


def test_first_prediction_depends_only_on_bos():
    """The prediction for the first token is the same for every sequence."""
    model = GPT(2, 64, vocab_size=2048)
    logits = model(torch.randint(0, 2048, (4, 64)))
    assert torch.allclose(logits[0, 0], logits[1:, 0], atol=1e-5)


@pytest.mark.parametrize("t", [0, 10, 63])
def test_no_lookahead(t):
    """Changing token t never changes the predictions for tokens 0..t."""
    torch.manual_seed(0)
    model = GPT(2, 64, vocab_size=2048)
    x = torch.randint(0, 2048, (1, 64))
    y = x.clone()
    y[0, t] = (y[0, t] + 1) % 2048
    a, b = model(x), model(y)
    assert torch.allclose(a[:, : t + 1], b[:, : t + 1], atol=1e-5)
    if t < 63:
        assert not torch.allclose(a[:, t + 1], b[:, t + 1], atol=1e-5)


def test_no_dropout():
    """In training mode the model is deterministic, so dropout is off."""
    model = GPT(2, 64, vocab_size=2048).train()
    x = torch.randint(0, 2048, (2, 64))
    assert torch.equal(model(x), model(x))


def test_bos_id_used():
    """The BOS id is what the model sees at the first position."""
    model = GPT(2, 64, vocab_size=50257, bos_id=50256)
    x = torch.randint(0, 50256, (2, 64))
    expected = model.gpt2(input_ids=torch.full((2, 1), 50256)).logits[:, 0]
    assert torch.allclose(model(x)[:, 0], expected, atol=1e-5)


def uniform_model(vocab_size=2048):
    """Return a GPT whose logits are all zero, i.e. a uniform distribution over tokens."""
    model = GPT(2, 64, vocab_size=vocab_size)
    torch.nn.init.zeros_(model.gpt2.transformer.wte.weight)
    return model


def test_bits_of_uniform_model():
    """A uniform model needs exactly 64 * log2(2048) = 704 bits per sequence."""
    x = torch.randint(0, 2048, (5, 64))
    assert np.allclose(uniform_model().bits(x), 704.0, atol=1e-3)


def test_bits_match_log_softmax():
    """Bits equal the summed negative log2-probability of every target token."""
    torch.manual_seed(0)
    model = GPT(2, 64, vocab_size=2048)
    x = torch.randint(0, 2048, (3, 64))
    logp = torch.log_softmax(model(x).detach().double(), dim=-1)
    expected = -logp.gather(2, x[..., None]).squeeze(-1).sum(1) / np.log(2)
    assert np.allclose(model.bits(x), expected.numpy(), atol=1e-3)


def test_bits_independent_of_batch_size():
    """Splitting the data into batches does not change the result."""
    model = GPT(2, 64, vocab_size=2048)
    x = np.random.default_rng(0).integers(0, 2048, size=(20, 64))
    assert np.allclose(model.bits(x, batch_size=3), model.bits(x, batch_size=100))


def test_bits_shape_and_dtype():
    """Bits come back as one float64 value per sequence."""
    x = np.random.default_rng(0).integers(0, 2048, size=(7, 64))
    bits = GPT(1, 32, vocab_size=2048).bits(x)
    assert bits.shape == (7,) and bits.dtype == np.float64


def test_pretrained_scores_like_gpt(tmp_path):
    """A saved model reloaded as PretrainedGPT gives the same bits as the original."""
    model = GPT(2, 64, vocab_size=2048, bos_id=0)
    model.gpt2.save_pretrained(tmp_path)
    x = torch.randint(0, 2048, (4, 64))
    assert np.allclose(PretrainedGPT(tmp_path, bos_id=0).bits(x), model.bits(x), atol=1e-4)


def test_pretrained_loads_fp32_eval(tmp_path):
    """Checkpoints saved in bf16 with dropout still load as fp32 in eval mode."""
    gpt2 = GPT2LMHeadModel(GPT2Config(n_layer=1, n_embd=32, n_head=1, n_positions=64))
    gpt2.to(torch.bfloat16).save_pretrained(tmp_path)
    reference = PretrainedGPT(tmp_path)
    assert next(reference.parameters()).dtype == torch.float32
    assert not reference.gpt2.training
    x = torch.randint(0, 50256, (2, 64))
    assert np.array_equal(reference.bits(x), reference.bits(x))


def test_pretrained_defaults_to_endoftext_bos(tmp_path):
    """The reference model conditions on <|endoftext|>, the GPT-2 document separator."""
    GPT(1, 32, vocab_size=50257).gpt2.save_pretrained(tmp_path)
    assert PretrainedGPT(tmp_path).bos_id == 50256


def test_pretrained_ignores_flash_attention_config(tmp_path):
    """A checkpoint whose config asks for FlashAttention 2 still loads, using SDPA instead."""
    GPT(1, 32, vocab_size=50257).gpt2.save_pretrained(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    config["_attn_implementation"] = "flash_attention_2"
    config["attn_implementation"] = "flash_attention_2"
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert PretrainedGPT(tmp_path).gpt2.config._attn_implementation == "sdpa"
