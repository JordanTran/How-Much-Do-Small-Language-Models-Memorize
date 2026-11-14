import torch
from transformers import GPT2Config, GPT2LMHeadModel


class GPT(torch.nn.Module):
    def __init__(self, n_layer, d_model, vocab_size, seq_len=64, bos_id=0, head_dim=32):
        """Build a randomly initialized GPT-2 with no dropout."""
        super().__init__()
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=seq_len,
            n_embd=d_model,
            n_layer=n_layer,
            n_head=d_model // head_dim,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.gpt2 = GPT2LMHeadModel(config)
        self.bos_id = bos_id

    def forward(self, x):
        """Return logits predicting every token of x from BOS and the tokens before it."""
        bos = torch.full_like(x[:, :1], self.bos_id)
        return self.gpt2(input_ids=torch.cat([bos, x[:, :-1]], dim=1)).logits

    def n_params(self):
        """Return the parameter count, counting the tied embedding once."""
        return sum(p.numel() for p in self.parameters())
